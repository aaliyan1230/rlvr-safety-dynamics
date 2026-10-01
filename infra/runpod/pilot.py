#!/usr/bin/env python3
"""Run a bounded, two-pod persistence/inference pilot, then delete pilot resources."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from runpod_cli import USER_AGENT, ApiError, list_pods, public_pod, request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
GPU = 'NVIDIA A100-SXM4-80GB'
RATE = 1.59
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
REMOTE = '/workspace/rlvr-safety-dynamics/pilot'


def save(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def emit(stage, **fields):
    print(json.dumps({'stage': stage, **fields}), flush=True)


def deleted(path):
    try:
        request(path)
    except ApiError as exc:
        if exc.status == 404:
            return True
        raise
    return False


def terminate(pod_id):
    path = '/pods/' + pod_id
    try:
        request(path, method='DELETE')
    except ApiError as exc:
        if exc.status != 404:
            raise
    for _ in range(12):
        if deleted(path):
            return
        time.sleep(5)
    raise RuntimeError('Pod deletion not verified: ' + pod_id)


def cleanup(state):
    # Match the unique run name too: POST may have succeeded before its reply was lost.
    ids = set(state['pod_ids'])
    ids.update(p['id'] for p in list_pods() if p.get('name', '').startswith(state['name'] + '-'))
    for pod_id in ids:
        terminate(pod_id)
    volumes = request('/network-volumes')['networkVolumes']
    for volume in volumes:
        if volume['name'] == state['name'] or volume['id'] == state.get('volume_id'):
            path = '/network-volumes/' + volume['id']
            try:
                request(path, method='DELETE')
            except ApiError as exc:
                if exc.status != 404:
                    raise
            if not deleted(path):
                raise RuntimeError('Volume deletion not verified')


def watchdog(path):
    while True:
        state = json.loads(path.read_text())
        if state.get('cleanup_verified'):
            return
        if time.time() >= state['deadline']:
            try:
                cleanup(state)
                emit('deadline_cleanup_verified')
                return
            except (ApiError, RuntimeError, OSError) as exc:
                emit('deadline_cleanup_retry', error=str(exc))
        time.sleep(10)


def ssh_command(connection, private_key, known_hosts):
    if not connection or not connection.get('host') or not connection.get('port'):
        raise RuntimeError('Direct SSH connection not ready')
    return ['ssh', '-i', str(private_key), '-p', str(connection['port']),
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'UserKnownHostsFile=' + str(known_hosts),
            connection.get('username', 'root') + '@' + connection['host']]


def remote(connection, args, command, *, input_text=None, timeout=45, check=True):
    result = subprocess.run(
        ssh_command(connection, args.private_key, args.output / 'known_hosts') + [command],
        input=input_text, text=True, capture_output=True, timeout=timeout, check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f'SSH command failed with exit {result.returncode}: {result.stderr[-500:]}')
    return result


def create_pod(args, state, path, number):
    body = {
        'name': state['name'] + '-' + str(number), 'cloud': 'SECURE', 'image': IMAGE,
        'gpu': {'id': GPU, 'count': 1, 'minCudaVersion': '12.8'},
        'dataCenterIds': [args.datacenter], 'disk': 30, 'ports': ['22/tcp'],
        'startSsh': True, 'startJupyter': False,
        'mounts': {'network': [{'volumeId': state['volume_id'], 'path': '/workspace'}]},
    }
    # Never retry a create automatically; recover via unique name and inventory.
    pod = request('/pods', method='POST', body=body)
    state['pod_ids'].append(pod['id'])
    state.setdefault('pods', []).append(public_pod(pod))
    save(path, state)
    emit('created', **public_pod(pod))
    if pod.get('cost', RATE + 1) > RATE + 0.10:
        raise RuntimeError('Accepted price exceeds pilot ceiling; terminating')
    until = min(time.time() + 600, state['deadline'])
    while time.time() < until:
        pod = request('/pods/' + pod['id'])
        emit('readiness', id=pod['id'], status=pod['status'])
        if pod['status'] == 'ERROR':
            raise RuntimeError('Pod entered ERROR')
        connection = (pod.get('ssh') or {}).get('direct')
        if connection:
            attempt = remote(connection, args, 'true', check=False)
            if attempt.returncode == 0:
                state['pods'][-1] = public_pod(pod)
                save(path, state)
                return pod, connection
        time.sleep(10)
    raise RuntimeError('SSH readiness deadline expired')


def bootstrap():
    return f'''set -eu
BASE=/workspace/rlvr-safety-dynamics
mkdir -p "$BASE/pilot" "$BASE/hf_cache" "$BASE/venvs"
if [ ! -d "$BASE/repo/.git" ]; then
 git -c http.userAgent='{USER_AGENT}' clone --depth 1 \
   https://github.com/aaliyan1230/rlvr-safety-dynamics.git "$BASE/repo"
fi
python3 -m venv --system-site-packages "$BASE/venvs/pilot"
export HF_HOME="$BASE/hf_cache" HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
"$BASE/venvs/pilot/bin/python" - <<'INSTALL'
from pip._vendor import requests
original = requests.Session.send
def send(self, request, **kwargs):
 request.headers['User-Agent'] = '{USER_AGENT}'
 return original(self, request, **kwargs)
requests.Session.send = send
from pip._internal.cli.main import main
raise SystemExit(main(['install', '--disable-pip-version-check',
 'transformers==4.57.1', 'huggingface-hub==0.34.4', 'accelerate==1.10.1']))
INSTALL
"$BASE/venvs/pilot/bin/python" "$BASE/pilot/pilot_workload.py"
'''


def run(args):
    if not args.yes:
        raise ValueError('Pilot requires explicit authorization and --yes')
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.chmod(0o700)
    path = args.output / 'state.json'
    state = {'name': 'rlvr-pilot-' + uuid.uuid4().hex[:10], 'pod_ids': [],
             'deadline': time.time() + 3600, 'started_at': time.time(),
             'gpu': GPU, 'rate_ceiling': RATE + 0.10, 'image': IMAGE,
             'datacenter': args.datacenter}
    save(path, state)
    with (args.output / 'watchdog.log').open('w') as log:
        guard = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                  'watchdog', '--state', str(path)],
                                 stdout=log, stderr=log, start_new_session=True)
    state['watchdog_pid'] = guard.pid
    save(path, state)
    try:
        catalog = request('/catalog/gpus', {
            'include': 'AVAILABILITY', 'product': 'POD', 'cloud': 'SECURE', 'count': 1})
        gpu = next(g for g in catalog['gpus'] if g['id'] == GPU)
        if gpu['price']['secure'] > RATE or not any(
            d['id'] == args.datacenter and d['availability'] != 'NONE'
            for d in gpu.get('dataCenters', [])
        ):
            raise RuntimeError('Approved GPU/rate/location unavailable')
        keys = request('/account/ssh-keys')['keys']
        public_key = args.private_key.with_suffix('.pub').read_text().strip()
        if public_key not in keys:
            updated = request('/account/ssh-keys', method='PUT', body={'keys': keys + [public_key]})
            if public_key not in updated['keys']:
                raise RuntimeError('SSH key registration not confirmed')
        volume = request('/network-volumes', method='POST', body={
            'name': state['name'], 'dataCenter': args.datacenter, 'size': 50, 'type': 'STANDARD'})
        state['volume_id'] = volume['id']
        state['volume'] = volume
        save(path, state)
        pod, connection = create_pod(args, state, path, 1)
        remote(connection, args, 'mkdir -p ' + REMOTE)
        remote(connection, args, 'cat > ' + REMOTE + '/pilot_workload.py',
               input_text=(HERE / 'pilot_workload.py').read_text())
        remote(connection, args, 'cat > ' + REMOTE + '/bootstrap.sh', input_text=bootstrap())
        launch = (f"nohup bash -c 'timeout 1800 bash {REMOTE}/bootstrap.sh "
                  f"> {REMOTE}/job.log 2>&1; echo $? > {REMOTE}/exit.code' "
                  "</dev/null >/dev/null 2>&1 & echo $!")
        emit('detached_job', pid=remote(connection, args, launch).stdout.strip())
        until = min(time.time() + 1860, state['deadline'])
        while time.time() < until:
            # Each call opens a new SSH session: workload survival is independent of it.
            progress = remote(connection, args,
                              f'tail -n 2 {REMOTE}/progress.jsonl 2>/dev/null; '
                              f'if [ -f {REMOTE}/exit.code ]; then cat {REMOTE}/exit.code; fi',
                              check=False)
            emit('progress', output=progress.stdout[-2000:])
            exit_code = remote(connection, args,
                               f'cat {REMOTE}/exit.code 2>/dev/null', check=False)
            if exit_code.returncode == 0:
                if exit_code.stdout.strip() != '0':
                    log = remote(connection, args, f'tail -n 50 {REMOTE}/job.log', check=False)
                    emit('job_failed', log=log.stdout)
                    raise RuntimeError('Pilot workload failed: ' + exit_code.stdout.strip())
                break
            time.sleep(15)
        else:
            raise RuntimeError('Workload deadline expired')
        result_text = remote(connection, args, f'cat {REMOTE}/result.json').stdout
        result = json.loads(result_text)
        if not result.get('passed'):
            raise RuntimeError('Workload did not confirm success')
        digest = hashlib.sha256(result_text.encode()).hexdigest()
        for filename in ['result.json', 'result.sha256', 'progress.jsonl', 'job.log']:
            content = remote(connection, args, f'cat {REMOTE}/{filename}').stdout
            (args.output / filename).write_text(content)
        terminate(pod['id'])
        state['first_pod_deleted_at'] = time.time()
        save(path, state)
        emit('first_pod_deleted', id=pod['id'])
        replacement, connection = create_pod(args, state, path, 2)
        persisted = remote(connection, args, f'cat {REMOTE}/result.json').stdout
        if hashlib.sha256(persisted.encode()).hexdigest() != digest:
            raise RuntimeError('Result did not survive Pod deletion')
        failure = remote(connection, args, 'bash -c "exit 7"', check=False)
        if failure.returncode != 7:
            raise RuntimeError('Intentional failure was not detected')
        state['checks'] = {'inference': True, 'detached_progress_polling': True,
                           'replacement_volume_sha256': digest, 'failure_exit': 7}
        state['passed'] = True
        emit('pilot_passed', **state['checks'])
    finally:
        cleanup(state)
        state['cleanup_verified'] = True
        state['finished_at'] = time.time()
        save(path, state)
        emit('cleanup_verified', elapsed_seconds=state['finished_at'] - state['started_at'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    pilot = subs.add_parser('run')
    pilot.add_argument('--yes', action='store_true')
    pilot.add_argument('--datacenter', default='EUR-IS-1')
    pilot.add_argument('--private-key', type=Path, required=True)
    pilot.add_argument('--output', type=Path, required=True)
    guard = subs.add_parser('watchdog')
    guard.add_argument('--state', type=Path, required=True)
    recovery = subs.add_parser('cleanup')
    recovery.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'watchdog':
        watchdog(args.state)
    elif args.command == 'cleanup':
        cleanup(json.loads(args.state.read_text()))
    else:
        run(args)


if __name__ == '__main__':
    main()
