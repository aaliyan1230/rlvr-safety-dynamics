"""Build a checksummed, credential-free execution bundle for a fixed checkpoint batch."""
from __future__ import annotations

import io
import json
import re
import shutil
import tarfile
from pathlib import Path

from .checkpoint_protocol import select_questions
from .checkpoint_study import atomic_json, digest
from .io import read_jsonl, sha256_file, write_jsonl
from .provenance import verify_manifest


def bootstrap_script() -> str:
    return '''#!/bin/bash
set -eu
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PWD/src"
python3.12 - <<'VERSION'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Locked runtime requires Python 3.12; refuse before model download')
VERSION
python3.12 -m venv --system-site-packages venv
venv/bin/python - <<'INSTALL'
from pip._vendor import requests
original = requests.Session.send
def send(self, request, **kwargs):
    request.headers['User-Agent'] = 'OpenAI File Downloader, XaiImageApiFetch/1.0'
    return original(self, request, **kwargs)
requests.Session.send = send
from pip._internal.cli.main import main
raise SystemExit(main(['install', '--disable-pip-version-check', '--require-hashes',
                      '--no-deps', '-r', 'runtime.lock']))
INSTALL
venv/bin/python checkpoint_workload.py --bundle "$PWD"
'''


def build_bundle(root: Path, prepared: Path, destination: Path, *, stage: str,
                 image: str, gpu: str, steps: list[int] | None = None,
                 cap: int = 8192, per_response_seconds: int = 600,
                 campaign_deadline: float, seed_records: list[dict] | None = None) -> dict:
    if cap not in (8192, 32768):
        raise ValueError('only the initial cap or single permitted repair cap is allowed')
    if not 1 <= per_response_seconds <= 1800:
        raise ValueError('response timeout must remain bounded within the batch lifetime')
    if not image.startswith('runpod/pytorch@sha256:') or not gpu:
        raise ValueError('pin the immutable image and selected GPU before building')
    if destination.exists():
        raise ValueError('bundle destination already exists; do not overwrite provenance')
    matrix = list(read_jsonl(prepared / 'matrix.jsonl'))
    lineages = {r['model'].split('_', 1)[0] for r in matrix}
    if len(lineages) != 1 or not lineages <= {'think', 'instruct'}:
        raise ValueError('prepared matrix needs exactly one supported model lineage')
    lineage = lineages.pop()
    selected = select_questions(matrix, stage, steps)
    metadata = json.loads((prepared / 'model_metadata.json').read_text())
    by_step = {m['step']: m for m in metadata}
    selected_steps = sorted({r['step'] for r in selected})
    full_pins = dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s\\]+)',
                               (root / 'infra/runpod/checkpoint-runtime.lock').read_text(), re.M))
    if not full_pins or full_pins.get('torch') != '2.8.0':
        raise ValueError('missing resolved runtime lock')
    template_sha = sha256_file(prepared / 'tokenizer/chat_template.jinja')
    source_files = sorted((root / 'src/rlvr_safety').rglob('*.py'))
    source_hashes = {str(p.relative_to(root)): sha256_file(p) for p in source_files}
    source_hashes['infra/runpod/checkpoint_workload.py'] = sha256_file(
        root / 'infra/runpod/checkpoint_workload.py')
    protocol = {'schema_version': 1, 'lineage': f'olmo-3-7b-{lineage}', 'precision': 'bf16',
                'attention_implementation': 'eager', 'trust_remote_code': False,
                'generation': {'max_new_tokens': cap, 'seed': 0, 'do_sample': False,
                               'per_response_seconds': per_response_seconds},
                'image': image, 'gpu': gpu, 'runtime_pins': full_pins,
                'runtime_lock_sha256': sha256_file(root / 'infra/runpod/checkpoint-runtime.lock'),
                'models': metadata, 'matrix_sha256': sha256_file(prepared / 'matrix.jsonl'),
                'source_hashes': source_hashes, 'native_template_sha256': template_sha,
                'tokenizer_sha256': {p.name: sha256_file(p)
                    for p in (prepared / 'tokenizer').iterdir()
                    if p.name in (
                        'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')},
                'written_protocol_sha256': sha256_file(
                    root / 'data/tulu_freeform_scoring_protocol_v1.md'),
                'claim_margin': .10, 'bootstrap_repetitions': 5000,
                'bootstrap_seed': 20260713, 'source_units': 24}
    destination.mkdir(parents=True)
    for source in source_files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(root / 'infra/runpod/checkpoint_workload.py',
                    destination / 'checkpoint_workload.py')
    shutil.copyfile(root / 'infra/runpod/checkpoint-runtime.lock', destination / 'runtime.lock')
    (destination / 'bootstrap.sh').write_text(bootstrap_script())
    atomic_json(destination / 'protocol.json', protocol)
    atomic_json(destination / 'batch.json', {'stage': stage, 'steps': selected_steps,
                                           'campaign_deadline': campaign_deadline})
    atomic_json(destination / 'model_metadata.json', [by_step[s] for s in selected_steps])
    write_jsonl(destination / 'matrix.jsonl', selected)
    shutil.copyfile(prepared / 'matrix.jsonl', destination / 'full_matrix.jsonl')
    seeds = seed_records or []
    if any(r.get('simulation') or r.get('protocol_hash') != digest(protocol) for r in seeds):
        raise ValueError('seed outputs must be real and match the complete frozen protocol')
    write_jsonl(destination / 'seed_records.jsonl', seeds)
    files = {str(p.relative_to(destination)): sha256_file(p)
             for p in destination.rglob('*') if p.is_file()}
    manifest = {'schema_version': 1, 'files': files, 'protocol_hash': digest(protocol)}
    atomic_json(destination / 'bundle_manifest.json', manifest)
    verify_prepared_bundle(destination)
    archive_path = destination.with_suffix('.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name in sorted([*files, 'bundle_manifest.json']):
            data = (destination / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o600
            member.mtime = 0
            archive.addfile(member, io.BytesIO(data))
    return {'bundle': str(destination), 'archive_sha256': sha256_file(archive_path),
            'protocol_hash': digest(protocol), 'stage': stage, 'steps': selected_steps,
            'rows': len(selected), 'full_runtime_packages': len(full_pins)}


def verify_prepared_bundle(directory: Path) -> dict:
    manifest = verify_manifest(directory / 'bundle_manifest.json')
    protocol = json.loads((directory / 'protocol.json').read_text())
    generation = protocol['generation']
    if (generation['max_new_tokens'] not in (8192, 32768)
            or generation['do_sample'] is not False or generation['seed'] != 0
            or protocol['precision'] != 'bf16' or protocol['trust_remote_code'] is not False):
        raise ValueError('prepared generation differs from the accepted protocol')
    batch = json.loads((directory / 'batch.json').read_text())
    if digest(protocol) != manifest['protocol_hash']:
        raise ValueError('prepared protocol hash mismatch')
    for name, expected in protocol['source_hashes'].items():
        relative = 'checkpoint_workload.py' if name.startswith('infra/runpod/') else name
        if sha256_file(directory / relative) != expected:
            raise ValueError('bundled code differs from the frozen protocol')
    if sha256_file(directory / 'runtime.lock') != protocol['runtime_lock_sha256']:
        raise ValueError('bundled runtime lock differs from the protocol')
    if sha256_file(directory / 'full_matrix.jsonl') != protocol['matrix_sha256']:
        raise ValueError('bundled full input inventory differs from the protocol')
    selected = select_questions(list(read_jsonl(directory / 'full_matrix.jsonl')),
                                batch['stage'], batch['steps'])
    if list(read_jsonl(directory / 'matrix.jsonl')) != selected:
        raise ValueError('batch inputs differ from the frozen deterministic selection')
    by_step = {m['step']: m for m in protocol['models']}
    if json.loads((directory / 'model_metadata.json').read_text()) != [
        by_step[s] for s in batch['steps']
    ]:
        raise ValueError('batch model metadata differs from the protocol')
    return manifest
