from __future__ import annotations

import json
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rlvr_safety.checkpoint_generation import HFCheckpointProvider
from rlvr_safety.checkpoint_study import (
    ResponseStore,
    build_matrix,
    coverage_ids,
    digest,
    enforce_exposure,
    execute_questions,
    extract_final,
    score_record,
    verify_chunk,
)

ROOT = Path(__file__).resolve().parents[1]


class CheckpointStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = build_matrix(ROOT)

    def output(self, row, **changes):
        choice = next(k for k, v in row['score_key'].items() if v == 2)
        return {'id': row['id'], 'input_hash': row['input_hash'], 'protocol_hash': digest({}),
                'model_revision_resolved': 'a' * 40, 'raw_response': f'CHOICE: {choice}',
                'generated_tokens': 5, 'reasoning_open': False, 'stop': 'complete', **changes}

    def test_full_matrices_and_coverage_are_balanced(self):
        self.assertEqual(Counter(r['panel'] for r in self.matrix), {'mcq': 6912, 'written': 144})
        fallback = build_matrix(ROOT, 'instruct')
        self.assertEqual(Counter(r['panel'] for r in fallback), {'mcq': 5184, 'written': 120})
        ids = coverage_ids(self.matrix)
        selected = [r for r in self.matrix if r['step'] == 0 and r['question_id'] in ids]
        self.assertEqual(len({r['source_id'] for r in selected}), 24)
        self.assertEqual(sorted(Counter(r['category'] for r in selected).values()), [8] * 3)
        self.assertEqual(sorted(Counter(r['wording_id'] for r in selected).values()), [6] * 4)
        self.assertEqual(sorted(Counter(r['option_order'] for r in selected).values()), [4] * 6)

    def test_every_mapping_scores_semantics_rather_than_letter(self):
        for row in self.matrix:
            if row['panel'] == 'mcq':
                scored = score_record(row, self.output(row))
                self.assertEqual(scored['instrumental_score_0_2'], 2)

    def test_matrix_rejects_semantic_changes_hidden_inside_option_permutations(self):
        from rlvr_safety.choice import build_choice_prompt
        from rlvr_safety.io import read_jsonl, write_jsonl
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'data').mkdir()
            rows = list(read_jsonl(ROOT / 'data/choice_factorial_v1.jsonl'))
            rows[1]['options'][0]['text'] = 'Meaning changed under an unchanged score key.'
            rows[1]['prompt'] = build_choice_prompt(rows[1]['base_prompt'], rows[1]['options'])
            write_jsonl(root / 'data/choice_factorial_v1.jsonl', rows)
            (root / 'data/tulu_freeform_anchor_v1.jsonl').write_bytes(
                (ROOT / 'data/tulu_freeform_anchor_v1.jsonl').read_bytes())
            with self.assertRaisesRegex(ValueError, 'semantic option meanings'):
                build_matrix(root)

    def test_failed_coverage_cannot_be_bypassed_by_resume(self):
        questions = self.matrix[:48]
        bad_ids = {r['id'] for r in questions[:2]}

        def provider(row, protocol_hash):
            changes = {'raw_response': 'No strict choice.'} if row['id'] in bad_ids else {}
            return self.output(row, protocol_hash=protocol_hash, **changes)

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            with self.assertRaisesRegex(ValueError, 'coverage gate'):
                execute_questions(questions, {}, provider, directory, deadline=time.time() + 60)
            self.assertEqual(len(ResponseStore(directory, {}, questions).records), 24)
            resumed = Mock()
            with self.assertRaisesRegex(ValueError, 'resume cannot bypass'):
                execute_questions(questions, {}, resumed, directory, deadline=time.time() + 60)
            resumed.assert_not_called()

    def test_failure_seals_completed_tail_for_retrieval_before_cleanup(self):
        questions = self.matrix[:24]
        calls = 0

        def provider(row, protocol_hash):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise TimeoutError('simulated provider disconnect')
            return self.output(row, protocol_hash=protocol_hash)

        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            with self.assertRaises(TimeoutError):
                execute_questions(questions, {}, provider, directory, deadline=time.time() + 60)
            chunks = list((directory / 'chunks').glob('*/manifest.json'))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(len(verify_chunk(chunks[0].parent)), 3)

    def test_extraction_never_scores_reasoning_or_incomplete_answer(self):
        fixtures = [
            ('CHOICE: B\nREASON: ünicode', False, 'complete', 'CHOICE: B\nREASON: ünicode'),
            ('<think>CHOICE: A</think>CHOICE: C', False, 'complete', 'CHOICE: C'),
            ('CHOICE: A</think>CHOICE: C', True, 'complete', 'CHOICE: C'),
            ('CHOICE: A', True, 'complete', ''),
            ('<think>CHOICE: A', False, 'complete', ''),
            ('<think>x</think>', False, 'complete', ''),
            ('<think>x</think>CHOICE: A', True, 'complete', ''),
            ('</think><think>CHOICE: A', False, 'complete', ''),
            ('CHOICE: A', False, 'length', ''),
            ('CHOICE: A', False, 'timeout', ''),
            ('', False, 'complete', ''),
        ]
        for raw, opened, stop, expected in fixtures:
            with self.subTest(raw=raw, opened=opened, stop=stop):
                self.assertEqual(extract_final(raw, reasoning_open=opened, stop=stop)[0], expected)
        for raw in ('CHOICE: A\nCHOICE: B', 'I choose A', 'I cannot answer', ''):
            scored = score_record(self.matrix[0], self.output(self.matrix[0], raw_response=raw))
            self.assertEqual(scored['instrumental_score_0_2'], '')

    def test_resume_survives_partial_tail_without_losing_or_duplicating_answers(self):
        rows = self.matrix[:3]
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            store = ResponseStore(directory, {}, rows)
            store.append(self.output(rows[1]))  # Arrival order need not match planned order.
            with self.assertRaises(ValueError):
                store.require_complete()
            with store.path.open('ab') as handle:
                handle.write(b'{"unfinished":')
            recovered = ResponseStore(directory, {}, rows)
            self.assertEqual(set(recovered.records), {rows[1]['id']})
            self.assertEqual(len(list(directory.glob('partial-*.bin'))), 1)
            with self.assertRaises(ValueError):
                recovered.append(self.output(rows[1]))
            for row in (rows[0], rows[2]):
                recovered.append(self.output(row))
            recovered.require_complete()
            sealed = recovered.seal()
            self.assertEqual(sealed['records'], 3)
            self.assertEqual(len(ResponseStore(directory, {}, rows).records), 3)
            with self.assertRaises(ValueError):
                ResponseStore(directory, {'changed': True}, rows)
            changed = [{**rows[0], 'prompt': 'changed'}, *rows[1:]]
            with self.assertRaises(ValueError):
                ResponseStore(directory, {}, changed)

    def test_rejects_schema_provenance_and_derived_score_corruption(self):
        row = {**self.matrix[0], 'revision': 'a' * 40}
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            store = ResponseStore(directory, {}, [row])
            for changes in ({'model_revision_resolved': 'b' * 40}, {'input_hash': 'wrong'},
                            {'id': 'wrong'}, {'generated_tokens': True},
                            {'reasoning_open': 'false'}, {'stop': 'unknown'},
                            {'protocol_hash': 'wrong'}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    store.append(self.output(row, **changes))
            store.append(self.output(row))
            record = json.loads(store.path.read_text())
            record['instrumental_score_0_2'] = 0
            store.path.write_text(json.dumps(record) + '\n')
            with self.assertRaises(ValueError):
                ResponseStore(directory, {}, [row])
            self.assertIn('"instrumental_score_0_2": 0', store.path.read_text())
            (directory / 'manifest.json').unlink()
            with self.assertRaises(ValueError):
                ResponseStore(directory, {}, [row])

    def test_deadline_and_budget_limits_prevent_expansion(self):
        args = dict(deadline=100, remaining_gpu=60, remaining_api=5,
                    maximum_gpu=10, maximum_api=1, seconds=20, now=0)
        enforce_exposure(**args)
        for changes in ({'now': 90}, {'remaining_gpu': 9}, {'remaining_api': 0},
                        {'maximum_gpu': float('nan')}, {'remaining_gpu': -1},
                        {'deadline': float('nan')}, {'now': float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                enforce_exposure(**{**args, **changes})

    def test_real_adapter_preserves_reasoning_markers_and_excludes_only_native_eos(self):
        tokenizer = SimpleNamespace(decode=Mock(return_value='thinking</think>CHOICE: A'))
        adapter = HFCheckpointProvider.__new__(HFCheckpointProvider)
        adapter.revision = 'a' * 40
        adapter.provider = SimpleNamespace(
            tokenizer=tokenizer, eos_ids=[99], provenance={'precision': 'bf16'},
            respond=Mock(return_value={
                'generated_token_ids': [1, 2, 99], 'generated_tokens': 3,
                'rendered_prompt': 'user message\nassistant\n<think>',
                'stop_status': 'complete'}))
        row = {**self.matrix[0], 'revision': 'a' * 40}
        output = adapter(row, digest({}))
        tokenizer.decode.assert_called_once_with([1, 2], skip_special_tokens=False)
        self.assertTrue(output['reasoning_open'])
        self.assertEqual(score_record(row, output)['choice'], 'A')
        with self.assertRaisesRegex(ValueError, 'loaded checkpoint'):
            adapter({**row, 'revision': 'b' * 40}, digest({}))


if __name__ == '__main__':
    unittest.main()
