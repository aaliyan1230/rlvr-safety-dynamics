#!/usr/bin/env python3
"""Exercise the scientific data path on full matrices with explicit synthetic providers."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rlvr_safety.checkpoint_study import (  # noqa: E402
    INSTRUCT_STEPS,
    THINK_STEPS,
    atomic_json,
    build_matrix,
    digest,
    execute_questions,
    verify_chunk,
)
from rlvr_safety.io import read_csv  # noqa: E402
from rlvr_safety.trajectory_analysis import (  # noqa: E402
    analyze_checkpoint_schedule,
    analyze_trajectory,
)


def provider(mode):
    def generate(row, protocol_hash):
        semantic = 2 if mode == 'behavior' and row['step'] >= 150 else 1
        if row['panel'] == 'mcq':
            label = 'A' if mode == 'order' and row['step'] >= 150 else next(
                k for k, v in row['score_key'].items() if v == semantic)
            final = f'CHOICE: {label}\nREASON: synthetic fixture, not a model result.'
        else:
            final = f'Synthetic written fixture; semantic class {semantic}.'
        return {'id': row['id'], 'input_hash': row['input_hash'],
                'protocol_hash': protocol_hash, 'raw_response': 'CHOICE: C</think>\n' + final,
                'reasoning_open': True, 'stop': 'complete', 'generated_tokens': 30,
                'model_revision_resolved': 'a' * 40}
    return generate


def report_values(value):
    """A zero behavioral denominator can yield an infinite nuisance/behavior ratio."""
    if isinstance(value, float) and not math.isfinite(value):
        return 'undefined' if math.isnan(value) else ('+infinity' if value > 0 else '-infinity')
    if isinstance(value, dict):
        return {k: report_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [report_values(v) for v in value]
    return value


def compare_metrics(first, second, path=''):
    """Permit summation-order roundoff, while requiring every key and decision to match."""
    if isinstance(first, dict) and isinstance(second, dict):
        if first.keys() != second.keys():
            raise ValueError(f'legacy metric keys changed: {path}')
        return max((compare_metrics(first[k], second[k], f'{path}/{k}') for k in first),
                   default=0)
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            raise ValueError(f'legacy metric list length changed: {path}')
        return max((compare_metrics(a, b, f'{path}/{i}')
                    for i, (a, b) in enumerate(zip(first, second, strict=True))), default=0)
    if type(first) is float and type(second) is float:
        if not math.isclose(first, second, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f'legacy numeric metric changed beyond roundoff: {path}')
        return abs(first - second)
    if type(first) is not type(second) or first != second:
        raise ValueError(f'legacy metric changed: {path}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--lineage', choices=('think', 'instruct'), default='think')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'FAIL', 'synthetic_only': True, 'paid_inference': False,
              'bootstrap_repetitions': args.bootstrap}
    try:
        matrix = build_matrix(ROOT, args.lineage)
        steps = THINK_STEPS if args.lineage == 'think' else INSTRUCT_STEPS
        model_steps = {f'{args.lineage}_{s:04d}': s for s in steps}
        protocol = {'synthetic': True, 'matrix_hash': digest(matrix)}
        for mode in ('behavior', 'order'):
            all_rows = []
            for step in steps:
                questions = [r for r in matrix if r['step'] == step]
                directory = args.out / mode / f'{step:04d}'
                if step == 0 and not (directory / 'manifest.json').exists():
                    try:
                        execute_questions(questions, protocol, provider(mode), directory,
                                          deadline=time.time() + 600, interrupt_after=13)
                    except InterruptedError:
                        pass
                    with (directory / 'responses.jsonl').open('ab') as handle:
                        handle.write(b'{"partial":')
                store = execute_questions(questions, protocol, provider(mode), directory,
                                          deadline=time.time() + 600)
                chunks = sorted((directory / 'chunks').glob('*/manifest.json'))
                retrieved = args.out / 'retrieved' / mode / f'{step:04d}'
                retrieved.mkdir(parents=True, exist_ok=True)
                retrieved_rows = []
                for manifest in chunks:
                    destination = retrieved / manifest.parent.name
                    shutil.copytree(manifest.parent, destination, dirs_exist_ok=True)
                    retrieved_rows.extend(verify_chunk(destination))
                if len(retrieved_rows) != len(questions):
                    raise ValueError('retrieval omitted or duplicated completed answers')
                if {r['id'] for r in retrieved_rows} != set(store.records):
                    raise ValueError('retrieval identity mismatch')
                all_rows.extend(r for r in retrieved_rows if r['panel'] == 'mcq')
                written = [r for r in retrieved_rows if r['panel'] == 'written']
                # Deterministic stand-in exercises written transport only, not judge validity.
                atomic_json(directory / 'synthetic-written-scores.json', [
                    {'id': r['id'], 'score_0_2': 2 if mode == 'behavior' and step >= 150 else 1,
                     'synthetic': True} for r in written])
            metrics = analyze_checkpoint_schedule(all_rows, model_steps,
                                                  bootstrap_repetitions=args.bootstrap)
            atomic_json(args.out / f'{mode}-metrics.json', report_values(metrics))
            if metrics['h6']['supported'] != (mode == 'behavior'):
                raise ValueError('behavior/order distinction failed')
            if mode == 'order':
                first, last = metrics['checkpoints'][0], metrics['checkpoints'][-1]
                if first['measurement'] == last['measurement']:
                    raise ValueError('synthetic order sensitivity not detected')
            report[mode] = {'responses': len(matrix), 'structured_drift_gate': metrics['h6'],
                            'retrieval_verified': True, 'interruption_resume_verified': True}
        config = json.loads((ROOT / 'configs/experiments/tulu_grpo_trajectory_v1.json').read_text())
        model_steps = {'tulu_step_0000_dpo': 0, **{
            f"tulu_step_{c['step']:04d}_grpo": c['step'] for c in config['checkpoints']}}
        score_paths = sorted((ROOT / 'artifacts').glob('tulu_trajectory_wave_*/choice_scores.csv'))
        rows = [r for p in score_paths for r in read_csv(p)]
        reproduced = analyze_trajectory(rows, model_steps, bootstrap_repetitions=5000)
        reproduced = json.loads(json.dumps(reproduced, allow_nan=False))
        original_path = ROOT / 'artifacts/tulu_structured_trajectory_v1/metrics.json'
        original = json.loads(original_path.read_text())
        atomic_json(args.out / 'tulu-reproduced.json', reproduced)
        maximum_difference = compare_metrics(original, reproduced)
        report['legacy_tulu_numerical_match'] = True
        report['legacy_tulu_exact_match'] = reproduced == original
        report['legacy_tulu_max_absolute_difference'] = maximum_difference
        report['legacy_tulu_tolerance'] = {'absolute': 1e-9, 'relative': 1e-10}
        report['status'] = 'PASS_SCIENTIFIC_DATA_PATH'
        report['remaining_stage_b_requirements'] = [
            'Exercise disconnected SSH and deadline/retrieval faults through the cloud controller',
            'Exercise metered structured judge through real/simulated transport',
            'Verify complete prepared cloud bundle and locked runtime before Stage C']
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        atomic_json(args.out.parent / 'rehearsal-report.json', report)


if __name__ == '__main__':
    main()
