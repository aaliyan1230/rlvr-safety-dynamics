#!/usr/bin/env python3
"""Execute a prepared OLMo batch; simulations exercise the same durable data path."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

from rlvr_safety.checkpoint_bundle import verify_prepared_bundle
from rlvr_safety.checkpoint_generation import HFCheckpointProvider
from rlvr_safety.checkpoint_protocol import readiness_summary
from rlvr_safety.checkpoint_study import ResponseStore, atomic_json, execute_questions
from rlvr_safety.io import read_jsonl
from rlvr_safety.permission.generation import save_runtime


def verify_runtime(protocol: dict) -> dict:
    observed = {name: importlib.metadata.version(name) for name in protocol['runtime_pins']}
    for name, version in observed.items():
        if version.split('+')[0] != protocol['runtime_pins'][name]:
            raise ValueError(f'locked package differs: {name}')
    if sys.version_info[:2] != (3, 12):
        raise ValueError('locked workload requires Python 3.12')
    return observed


def simulate(row, protocol_hash, pattern='safe'):
    semantic = (2 if row['step'] >= 150 else 1) if pattern == 'behavior' else (
        1 if pattern == 'order' else 0)
    choice = row.get('expected_choice') or next(
        (k for k, v in row.get('score_key', {}).items() if v == semantic), 'A')
    if pattern == 'order' and row['step'] >= 150 and not row.get('engineering_control'):
        choice = 'A'
    final = (f'CHOICE: {choice}\nREASON: deterministic engineering fixture.'
             if row['panel'] == 'mcq' else f'Synthetic score: {semantic}. Fixture only.')
    return {'id': row['id'], 'input_hash': row['input_hash'], 'protocol_hash': protocol_hash,
            'model_revision_resolved': row['revision'], 'simulation': True,
            'simulation_pattern': pattern,
            'raw_response': 'Synthetic reasoning.</think>\n' + final, 'reasoning_open': True,
            'generated_tokens': 20, 'stop': 'complete', 'inference_seconds': .001}


def run_workload(root: Path, *, simulated: bool = False, interrupt_after: int | None = None,
                 simulation_pattern: str = 'safe'):
    if not simulated and simulation_pattern != 'safe':
        raise ValueError('synthetic pattern is only available in explicit simulation mode')
    verify_prepared_bundle(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    batch = json.loads((root / 'batch.json').read_text())
    deadline = batch['campaign_deadline']
    if deadline <= time.time():
        raise TimeoutError('campaign deadline expired before any model load')
    if not simulated:
        verify_runtime(protocol)
    questions = list(read_jsonl(root / 'matrix.jsonl'))
    seeds = list(read_jsonl(root / 'seed_records.jsonl'))
    metadata = json.loads((root / 'model_metadata.json').read_text())
    output = root / 'out'
    output.mkdir(exist_ok=True)
    phase = {'stage': 'starting', 'simulation': simulated, 'completed': 0,
             'requested': len(questions)}
    done = threading.Event()

    def heartbeat():
        while not done.is_set():
            atomic_json(output / 'progress.json', {'time': time.time(), **phase})
            done.wait(10)

    guard = threading.Thread(target=heartbeat, daemon=True)
    guard.start()
    all_records = []
    summary = {'completed': False, 'simulation': simulated, 'models': {}}
    try:
        for model in metadata:
            checkpoint = [r for r in questions if r['step'] == model['step']]
            label = checkpoint[0]['model']
            phase.update(stage='model_load', model=label)
            directory = output / label
            directory.mkdir(exist_ok=True)
            config = {'model': {'repo': model['repo'], 'revision': model['revision'],
                                'label': label,
                                'chat_template_sha256': protocol['native_template_sha256']},
                      'generation': protocol['generation'],
                      'tokenizer_sha256': protocol['tokenizer_sha256'],
                      'runtime_pins': {k: v for k, v in protocol['runtime_pins'].items()
                                       if k in ('torch', 'transformers', 'huggingface-hub',
                                                'accelerate', 'safetensors', 'tokenizers',
                                                'sentencepiece', 'protobuf')},
                      'launch': {'image': protocol['image'], 'gpu': protocol['gpu']},
                      'readouts': ['generate']}
            provider = None
            try:
                if simulated:
                    atomic_json(directory / 'runtime.json', {'simulation': True,
                                                             'inference_performed': False})
                    def generate(row, protocol_hash):
                        return simulate(row, protocol_hash, simulation_pattern)
                else:
                    provider = HFCheckpointProvider(config, {
                        **model, 'id': model['repo'], 'sha': model['revision']}, root / 'hf_cache')
                    save_runtime(directory / 'runtime.json', provider.provider)
                    generate = provider
                store = ResponseStore(directory, protocol, checkpoint)
                for record in seeds:
                    if record['id'] in store.questions and record['id'] not in store.records:
                        store._validate(record)
                        store.append(record)

                def progress_provider(row, protocol_hash, generate=generate):
                    phase.update(stage='inference', id=row['id'])
                    response = generate(row, protocol_hash)
                    phase['completed'] += 1
                    return response

                if batch['stage'] == 'pilot' and not store.records:
                    try:
                        execute_questions(checkpoint, protocol, progress_provider, directory,
                            deadline=deadline, interrupt_after=5, enforce_health=False)
                    except InterruptedError:
                        recovered = ResponseStore(directory, protocol, checkpoint)
                        if len(recovered.records) != 5:
                            raise ValueError('pilot lost completed responses') from None
                        atomic_json(directory / 'recovery-proof.json', {
                            'interrupted_after': 5, 'durable_answers_recovered': 5,
                            'passed': True})
                store = execute_questions(checkpoint, protocol, progress_provider, directory,
                    deadline=deadline, interrupt_after=interrupt_after,
                    enforce_health=batch['stage'] != 'pilot')
                rows = list(store.records.values())
                all_records.extend(rows)
                summary['models'][label] = {'requested': len(checkpoint), 'saved': len(rows)}
            finally:
                if provider is not None:
                    provider.close()
                    cache = root / 'hf_cache'
                    if cache.exists():
                        shutil.rmtree(cache)
        health = readiness_summary(all_records, batch['stage'])
        atomic_json(output / 'readiness_generation.json', health)
        if not health['passed_generation']:
            raise ValueError('generation readiness checks failed; preserve evidence and stop')
        summary.update(completed=len(all_records) == len(questions),
                       records=len(all_records), generation_health=health)
    finally:
        atomic_json(output / 'run_summary.json', summary)
        done.set()
        guard.join()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--interrupt-after', type=int)
    parser.add_argument('--simulation-pattern', choices=('safe', 'behavior', 'order'),
                        default='safe')
    args = parser.parse_args()
    def terminate(signum, frame):
        raise TimeoutError('workload termination signal; seal completed answers')
    signal.signal(signal.SIGTERM, terminate)
    run_workload(args.bundle.resolve(), simulated=args.simulate,
                 interrupt_after=args.interrupt_after, simulation_pattern=args.simulation_pattern)


if __name__ == '__main__':
    main()
