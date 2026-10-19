#!/usr/bin/env python3
"""One supervised checkpoint batch with incremental retrieval and unconditional cleanup."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import permission_run
import pilot
from runpod_cli import ApiError, list_pods, public_pod, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from rlvr_safety.checkpoint_study import ResponseStore, atomic_json, verify_chunk  # noqa: E402
from rlvr_safety.io import read_jsonl, sha256_file  # noqa: E402
from rlvr_safety.provenance import verify_manifest  # noqa: E402


def export_command(root: str) -> str:
    """Only immutable completed chunks enter an incremental archive."""
    return f"""python3 - <<'EXPORT'
import io, tarfile
from pathlib import Path
root = Path({root!r}) / 'out'
stream = io.BytesIO()
with tarfile.open(fileobj=stream, mode='w:gz') as archive:
    if (root.parent / 'job.log').is_file():
        archive.add(root.parent / 'job.log', arcname='job.log', recursive=False)
    if root.exists():
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            sealed = path.parent.name.isdigit() and (path.parent / 'manifest.json').is_file()
            evidence = path.name in ('runtime.json', 'run_summary.json', 'health.json',
                                     'readiness_generation.json', 'recovery-proof.json')
            if sealed or evidence:
                archive.add(path, arcname=str(relative), recursive=False)
import sys
sys.stdout.buffer.write(stream.getvalue())
EXPORT
"""


def retrieve(connection, args, root: str) -> int:
    exported = permission_run.ssh(connection, args, export_command(root), timeout=120)
    target = args.output / 'retrieved'
    # Existing sealed data must never be overwritten with different remote bytes.
    staging = args.output / 'retrieval-staging'
    staging.mkdir(exist_ok=True)
    permission_run.extract_results(exported.stdout, staging)
    for manifest in staging.glob('*/chunks/*/manifest.json'):
        verify_chunk(manifest.parent)
    for path in staging.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(staging)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if 'chunks' in relative.parts and destination.exists():
            if sha256_file(destination) != sha256_file(path):
                raise ValueError('immutable retrieved evidence changed')
        else:
            destination.write_bytes(path.read_bytes())
    return len(list(target.glob('*/chunks/*/manifest.json')))


def verify_retrieved(args) -> int:
    questions = list(read_jsonl(args.bundle / 'matrix.jsonl'))
    protocol = json.loads((args.bundle / 'protocol.json').read_text())
    validator = ResponseStore(args.output / 'verified', protocol, questions)
    for manifest in sorted((args.output / 'retrieved').glob('*/chunks/*/manifest.json')):
        for record in verify_chunk(manifest.parent):
            if record.get('simulation'):
                raise ValueError('simulated answers cannot count as paid inference evidence')
            validator._validate(record)
            validator.records[record['id']] = record
    validator.require_complete()
    return len(validator.records)


def run_batch(args, report: dict, *, backup) -> dict:
    """Caller supplies a fresh passing preflight and a verified private backup callback."""
    if not report.get('ok'):
        raise ValueError('launch requires passing readiness and native preflight')
    if time.time() - report['checked_epoch'] > 120:
        raise ValueError('stale preflight; refresh native billing and inventory')
    if not args.private_key.is_file() or not args.private_key.with_suffix('.pub').is_file():
        raise ValueError('explicit existing local SSH keypair required before launch')
    launch = report['launch']
    if launch['max_seconds'] > 21600 or launch['max_seconds'] < 900:
        raise ValueError('invalid batch lifetime')
    if args.deadline - time.time() < launch['max_seconds']:
        raise ValueError('campaign deadline leaves insufficient time for batch cleanup')
    verify_manifest(args.bundle / 'bundle_manifest.json')
    archive = args.bundle.with_suffix('.tar.gz')
    permission_run.verify_archive(archive, args.bundle)
    archive_hash = sha256_file(archive)
    if report.get('bundle_sha256') != archive_hash:
        raise ValueError('preflight did not accept this exact bundle archive')
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.chmod(0o700)
    state_path = args.output / 'state.json'
    deadline = time.time() + launch['max_seconds']
    state = {'name': 'rlvr-olmo-' + uuid.uuid4().hex[:12], 'pod_ids': [],
             'started_at': time.time(), 'deadline': deadline, 'launch': launch,
             'cleanup_verified': False, 'passed': False, 'bundle_sha256': archive_hash}
    atomic_json(args.output / 'preflight.json', report)
    pilot.save(state_path, state)
    with (args.output / 'watchdog.log').open('w') as log:
        guard = subprocess.Popen([sys.executable, str(Path(__file__).with_name('pilot.py')),
                                  'watchdog', '--state', str(state_path)],
                                 stdout=log, stderr=log, start_new_session=True)
    state['watchdog_pid'] = guard.pid
    pilot.save(state_path, state)
    connection = None
    root = '/workspace/' + state['name']
    try:
        public_key = args.private_key.with_suffix('.pub').read_text().strip()
        keys = request('/account/ssh-keys')['keys']
        if public_key not in keys:
            result = request('/account/ssh-keys', method='PUT', body={'keys': keys + [public_key]})
            if public_key not in result['keys']:
                raise RuntimeError('public SSH key registration unconfirmed')
        pod = request('/pods', method='POST', body={
            'name': state['name'] + '-1', 'cloud': 'SECURE', 'image': launch['image'],
            'gpu': {'id': launch['gpu'], 'count': 1, 'minCudaVersion': '12.8'},
            'dataCenterIds': [launch['datacenter']], 'disk': launch['disk_gb'],
            'ports': ['22/tcp'], 'startSsh': True, 'startJupyter': False})
        state['pod_ids'].append(pod['id'])
        state['pod'] = public_pod(pod)
        pilot.save(state_path, state)
        if pod.get('cost', float('inf')) > launch['hourly_ceiling']:
            raise RuntimeError('accepted Pod quote exceeds ceiling')
        ready_until = min(deadline - 600, time.time() + 600)
        while time.time() < ready_until:
            pod = request('/pods/' + pod['id'])
            if pod['status'] == 'ERROR':
                raise RuntimeError('Pod entered ERROR')
            connection = (pod.get('ssh') or {}).get('direct')
            if connection and permission_run.ssh(
                connection, args, 'true', check=False
            ).returncode == 0:
                break
            time.sleep(10)
        else:
            raise TimeoutError('SSH readiness deadline expired')
        permission_run.ssh(connection, args, f'mkdir -p {root}; cat > {root}/bundle.tar.gz',
                           data=archive.read_bytes(), timeout=180)
        uploaded_hash = permission_run.ssh(
            connection, args, f'sha256sum {root}/bundle.tar.gz').stdout.decode().split()[0]
        if uploaded_hash != archive_hash:
            raise ValueError('uploaded archive checksum mismatch')
        permission_run.ssh(connection, args, permission_run.extract_bundle_command(root))
        # The prepared, hashed bootstrap owns the fully locked install and workload command.
        seconds = int(deadline - time.time() - 300)
        if seconds <= 0:
            raise TimeoutError('setup consumed workload/cleanup window')
        command = (f'timeout {seconds} bash {root}/bootstrap.sh > {root}/job.log 2>&1; '
                   f'echo $? > {root}/exit.code')
        launched = permission_run.ssh(connection, args,
            f'nohup bash -c {shlex.quote(command)} </dev/null >/dev/null 2>&1 & echo $!')
        state['workload_pid'] = launched.stdout.decode().strip()
        pilot.save(state_path, state)
        while time.time() < deadline - 240:
            result = permission_run.ssh(connection, args,
                f'cat {root}/exit.code 2>/dev/null', check=False)
            state['retrieved_chunks'] = retrieve(connection, args, root)
            pilot.save(state_path, state)
            if result.returncode == 0:
                state['workload_exit'] = int(result.stdout.decode().strip())
                if state['workload_exit']:
                    raise RuntimeError('workload failed; retrieved evidence retained')
                summary = json.loads((args.output / 'retrieved/run_summary.json').read_text())
                if not summary.get('completed'):
                    raise RuntimeError('workload reported incomplete data')
                state['verified_responses'] = verify_retrieved(args)
                provenance = args.output / 'retrieved/provenance'
                provenance.mkdir(exist_ok=True)
                (provenance / 'bundle.tar.gz').write_bytes(archive.read_bytes())
                backup(args.output / 'retrieved')
                state['backup_verified'] = True
                state['passed'] = True
                break
            time.sleep(15)
        else:
            raise TimeoutError('batch deadline; stop generation and preserve completed chunks')
    except BaseException as error:
        state['passed'] = False
        if isinstance(error, ApiError):
            state['api_error'] = {'status': error.status, 'problem': error.problem}
        raise
    finally:
        try:
            if connection:
                retrieve(connection, args, root)
        except BaseException:
            state['passed'] = False
            raise
        finally:
            pilot.save(state_path, state)
            try:
                pilot.cleanup(state)
            except BaseException:
                state['passed'] = False
                pilot.save(state_path, state)
                raise
            state['cleanup_verified'] = True
            state['finished_at'] = time.time()
            pilot.save(state_path, state)
            # Independently check native inventory; never delete another operator's resources.
            volumes = request('/network-volumes')['networkVolumes']
            remaining = {'pods': [p['id'] for p in list_pods()],
                         'volumes': [v['id'] for v in volumes]}
            atomic_json(args.output / 'inventory-after.json', remaining)
            if remaining['pods'] or remaining['volumes']:
                state['passed'] = False
                pilot.save(state_path, state)
                raise RuntimeError('native inventory is not empty after cleanup')
    return state
