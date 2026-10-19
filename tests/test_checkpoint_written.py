from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlvr_safety.checkpoint_analysis import written_comparisons
from rlvr_safety.checkpoint_judge import MeteredJudge, score_written
from rlvr_safety.freeform_review import blind_id


def rows():
    return [{'id': f'{step}-{source}', 'model': f'model-{step}', 'step': step,
             'source_id': str(source), 'category': 'fixture', 'panel': 'written',
             'prompt': 'Fixture question', 'response': f'fixture score={source % 3}',
             'final_status': 'complete', 'stop': 'complete',
             'score_anchors_0_2': {'0': 'permission', '1': 'mixed', '2': 'unauthorized'}}
            for step in (0, 150) for source in range(6)]


class WrittenPipelineTests(unittest.TestCase):
    def test_two_isolated_blind_passes_adjudication_and_versioned_budget_accounting(self):
        records = rows()
        disagree_id = blind_id(records[0], 'B')
        packets = []

        def transport(payload):
            packet = json.loads(payload['contents'][0]['parts'][0]['text'].split('\n\n', 1)[1])
            packets.append(packet)
            score = int(packet['response'].split('=')[1])
            if packet['blinded_row_id'] == disagree_id:
                score = (score + 1) % 3
            label = {'score_0_2_or_minus_1': score, 'judgeable': True, 'censor_reason': 'none',
                     'confidence': 'high', 'rationale': 'Deterministic fixture class.'}
            return {'modelVersion': 'fixture-version', 'usageMetadata': {
                'promptTokenCount': 100, 'candidatesTokenCount': 50, 'thoughtsTokenCount': 20},
                'candidates': [{'finishReason': 'STOP', 'content': {
                    'parts': [{'text': json.dumps(label)}]}}]}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            judge = MeteredJudge(root / 'judge', api_key='not-a-real-key', ceiling=5,
                                input_per_million=2, output_per_million=12, transport=transport)
            report = score_written(records, judge, root / 'scores')
            self.assertTrue(report['agreement_passed'])
            self.assertEqual(report['agreement']['adjudication_rows'], 1)
            self.assertEqual(report['judgeable'], 12)
            self.assertEqual(len(packets), 25)
            for packet in packets:
                self.assertEqual(set(packet) - {'adjudication'},
                                 {'blinded_row_id', 'prompt', 'response', 'score_anchors_0_2'})
            repeated = score_written(records, judge, root / 'scores')
            self.assertEqual(repeated, report)
            self.assertEqual(len(packets), 25)
            comparison = written_comparisons(report['labels'], repetitions=20)
            contrast = comparison['baseline_contrasts'][0]
            self.assertEqual(contrast['estimate'], 0)
            self.assertEqual(contrast['paired_sources'], 6)

    def test_missing_or_unjudgeable_written_answers_are_not_safe_zeros(self):
        labels = [{'id': r['id'], 'model': r['model'], 'checkpoint': r['step'],
                   'source_id': r['source_id'], 'judgeable': True, 'score_0_2': 1}
                  for r in rows()]
        labels[-1].update(judgeable=False, score_0_2=None)
        result = written_comparisons(labels, repetitions=20)
        contrast = result['baseline_contrasts'][0]
        self.assertEqual(contrast['paired_sources'], 5)
        self.assertLess(contrast['missingness_sensitivity_bounds'][0], 0)
        self.assertGreater(contrast['missingness_sensitivity_bounds'][1], 0)
        self.assertFalse(contrast['censoring_increase_within_five_pp'])
        with self.assertRaisesRegex(ValueError, 'missing or duplicated'):
            written_comparisons(labels[:-1], repetitions=20)
        with self.assertRaisesRegex(ValueError, 'unique'):
            written_comparisons([*labels, labels[0]], repetitions=20)


if __name__ == '__main__':
    unittest.main()
