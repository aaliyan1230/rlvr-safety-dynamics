from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rlvr_safety.checkpoint_judge import MeteredJudge, validate_label

LABEL = {'score_0_2_or_minus_1': 0, 'judgeable': True, 'censor_reason': 'none',
         'confidence': 'high', 'rationale': 'Explicitly requires operator permission.'}
PACKET = {'blinded_row_id': 'anonymous', 'prompt': 'What should the agent do?',
          'response': 'Ask the operator first.', 'score_anchors_0_2': {'0': 'permission'}}


def response(**changes):
    return {'modelVersion': 'fixed-version', 'usageMetadata': {
        'promptTokenCount': 300, 'candidatesTokenCount': 50, 'thoughtsTokenCount': 30},
        'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
            {'text': 'private thinking is not the answer', 'thought': True},
            {'text': json.dumps(LABEL)}]}}], **changes}


class MeteredJudgeTests(unittest.TestCase):
    def judge(self, path, transport, **changes):
        return MeteredJudge(path, api_key='test-not-a-real-key', ceiling=5,
                            input_per_million=2, output_per_million=12,
                            transport=transport, **changes)

    def test_structured_usage_version_pin_and_idempotent_completed_retry(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)
            transport = Mock(return_value=response())
            judge = self.judge(path, transport)
            result = judge.review(PACKET, reviewer='A')
            self.assertEqual(result['label'], LABEL)
            self.assertEqual(judge.version, 'fixed-version')
            self.assertAlmostEqual(judge.exposure(), .00156)
            self.assertEqual(judge.review(PACKET, reviewer='A'), result)
            transport.assert_called_once()
            resumed = self.judge(path, transport)
            self.assertEqual(resumed.review(PACKET, reviewer='A'), result)
            transport.assert_called_once()
            transport.return_value = response(modelVersion='changed-version')
            with self.assertRaisesRegex(ValueError, 'version changed'):
                resumed.review(PACKET, reviewer='B')

    def test_reserves_ambiguous_attempts_and_bounds_retries_before_calling(self):
        with tempfile.TemporaryDirectory() as td:
            transport = Mock(side_effect=TimeoutError('simulated lost response'))
            judge = self.judge(Path(td), transport)
            for _ in range(2):
                with self.assertRaises(TimeoutError):
                    judge.review(PACKET, reviewer='A')
            self.assertGreater(judge.exposure(), 0)
            with self.assertRaisesRegex(ValueError, 'attempts exhausted'):
                judge.review(PACKET, reviewer='A')
            self.assertEqual(transport.call_count, 2)
            judge.ceiling = judge.exposure()
            with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                judge.review(PACKET, reviewer='B')
            self.assertEqual(transport.call_count, 2)

    def test_schema_censoring_usage_and_blinding_faults_never_produce_scores(self):
        bad_responses = [response(usageMetadata=None), response(modelVersion=''),
                         response(candidates=[{'finishReason': 'MAX_TOKENS'}]),
                         response(candidates=[{'finishReason': 'STOP', 'content': {
                             'parts': [{'text': '{malformed'}]}}]),
                         response(usageMetadata={'promptTokenCount': -1,
                                                 'candidatesTokenCount': 2})]
        for bad in bad_responses:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as td:
                judge = self.judge(Path(td), Mock(return_value=bad))
                with self.assertRaises((ValueError, TypeError)):
                    judge.review(PACKET, reviewer='A')
                self.assertFalse(list(Path(td).glob('completed-*.json')))
        with tempfile.TemporaryDirectory() as td:
            transport = Mock()
            judge = self.judge(Path(td), transport)
            with self.assertRaisesRegex(ValueError, 'unblinded'):
                judge.review({**PACKET, 'checkpoint': 150}, reviewer='A')
            transport.assert_not_called()
        for changes in ({'score_0_2_or_minus_1': True}, {'judgeable': False},
                        {'censor_reason': 'token_capped'}, {'rationale': ''},
                        {'score_0_2_or_minus_1': 3}, {'confidence': 'invented'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_label({**LABEL, **changes})


if __name__ == '__main__':
    unittest.main()
