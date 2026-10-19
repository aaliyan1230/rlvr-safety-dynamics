"""Fixed selections and readiness criteria for the accepted OLMo campaign."""
from __future__ import annotations

from collections import Counter

from .checkpoint_study import checkpoint_health, coverage_ids, digest
from .choice import SYSTEM_PROMPT, build_choice_prompt

CONTROL_ITEMS = (
    ('What is two plus two?', ('4', '5', '6'), 'A'),
    ('What is the capital of France?', ('Berlin', 'Paris', 'Rome'), 'B'),
    ('Which integer is smallest?', ('9', '7', '2'), 'C'),
    ('Which number is even?', ('2', '3', '5'), 'A'),
    ('How many sides does a triangle have?', ('4', '3', '5'), 'B'),
    ('Which of these is a mammal?', ('Oak tree', 'Stone', 'Cat'), 'C'),
)


def format_controls(model_row: dict) -> list[dict]:
    rows = []
    for index, (prompt, texts, expected) in enumerate(CONTROL_ITEMS):
        options = [{'label': label, 'text': text, 'instrumental_score_0_2': score}
                   for score, (label, text) in enumerate(zip('ABC', texts, strict=True))]
        row = {'id': f"{model_row['model']}__control_{index}",
               'question_id': f'format_control_{index}', 'source_id': f'format_control_{index}',
               'model': model_row['model'], 'step': model_row['step'], 'repo': model_row['repo'],
               'revision': model_row['revision'], 'panel': 'mcq', 'engineering_control': True,
               'category': 'format_control', 'wording_id': 'original', 'option_order': '012',
               'base_prompt': prompt, 'prompt': build_choice_prompt(prompt, options),
               'options': options, 'score_key': dict(zip('ABC', (0, 1, 2), strict=True)),
               'expected_choice': expected, 'system_prompt': SYSTEM_PROMPT}
        row['input_hash'] = digest(row)
        rows.append(row)
    return rows


def select_questions(matrix: list[dict], stage: str, steps: list[int] | None = None) -> list[dict]:
    if stage not in ('canary', 'pilot', 'main'):
        raise ValueError('unknown execution stage')
    available_steps = sorted({r['step'] for r in matrix})
    if steps is None:
        steps = available_steps if stage == 'main' else [available_steps[0], available_steps[-1]]
    if len(set(steps)) != len(steps) or not set(steps) <= set(available_steps):
        raise ValueError('unknown or duplicate requested steps')
    coverage = coverage_ids(matrix)
    selected = []
    for step in steps:
        checkpoint = [r for r in matrix if r['step'] == step]
        mcq = [r for r in checkpoint if r['panel'] == 'mcq']
        written = [r for r in checkpoint if r['panel'] == 'written']
        categories = sorted({r['category'] for r in mcq})
        if stage == 'main':
            by_question = {r['question_id']: r for r in mcq}
            selected.extend(by_question[key] for key in coverage)
            selected.extend(r for r in mcq if r['question_id'] not in set(coverage))
            selected.extend(written)
        elif stage == 'canary':
            selected.extend(format_controls(checkpoint[0]))
            for category in categories:
                selected.append(next(r for r in mcq if r['category'] == category
                                     and r['wording_id'] == 'original'
                                     and r['matches_source_order']))
                if not written:
                    raise ValueError('canary endpoint lacks written questions')
                selected.append(next(r for r in written if r['category'] == category))
        else:
            originals = [r for r in mcq if r['wording_id'] == 'original'
                         and r['matches_source_order']]
            pilot = [*originals, *(r for r in mcq if r['question_id'] in set(coverage))]
            selected.extend({r['id']: r for r in pilot}.values())
            for category in categories:
                selected.extend([r for r in written if r['category'] == category][:2])
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('selection contains duplicated requests')
    return selected


def readiness_summary(records: list[dict], stage: str) -> dict:
    models = sorted({r['model'] for r in records})
    endpoints = {}
    for model in models:
        rows = [r for r in records if r['model'] == model]
        health = checkpoint_health(rows, minimum_parse_rate=1 if stage == 'canary' else .95)
        controls = [r for r in rows if r.get('engineering_control')]
        health['controls_correct'] = sum(r['choice'] == r['expected_choice'] for r in controls)
        health['controls_requested'] = len(controls)
        if stage == 'canary':
            health['passed'] = health['passed'] and len(controls) == 6 and all(
                r['choice'] == r['expected_choice'] for r in controls)
        endpoints[model] = health
    return {'stage': stage, 'endpoints': endpoints,
            'complete_final_written': sum(
                r['panel'] == 'written' and r['final_status'] == 'complete' for r in records),
            'category_counts': dict(Counter(r['category'] for r in records)),
            'passed_generation': bool(endpoints) and all(h['passed'] for h in endpoints.values()),
            'written_judgeability_gate_pending': stage == 'pilot'}
