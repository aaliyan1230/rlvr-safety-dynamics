from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlvr_safety.checkpoint_study import atomic_json, digest
from rlvr_safety.io import sha256_file

SPEC = importlib.util.spec_from_file_location(
    'olmo_campaign', Path(__file__).resolve().parents[1] / 'scripts/olmo_campaign.py')
campaign_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign_cli)


class RemainingMainForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = {'generation': {'max_new_tokens': 8192},
                         'gpu': 'NVIDIA H100 80GB HBM3', 'image': 'pinned-image'}
        (self.root / 'evidence.txt').write_text('measured fixture')
        self.forecast = {
            'status': 'PASS', 'created_epoch': 1000,
            'protocol_hash': digest(self.protocol),
            'protocol_fingerprint': campaign_cli.checkpoint_preflight.readiness_fingerprint(
                self.protocol),
            'evidence': {'evidence.txt': sha256_file(self.root / 'evidence.txt')},
            'forecast_fits_with_50_percent_contingency': True,
            'gpu_seconds_with_50_percent_contingency': 3000,
            'reserved_finalization_seconds': 600,
            'gpu_forecast_with_50_percent_contingency_usd': 20,
            'api_charged_or_reserved_so_far_usd': 1,
            'api_remaining_with_50_percent_contingency_usd': 2}
        atomic_json(self.root / 'main-freeze.json', self.forecast)
        atomic_json(self.root / 'campaign.json', {
            'deadline': datetime.fromtimestamp(10000, UTC).isoformat(),
            'gpu_ceiling_usd': 300, 'api_ceiling_usd': 25})

    def guard(self, **kwargs):
        campaign_cli.enforce_remaining_main_forecast(
            self.root, self.protocol, 10000, **kwargs)

    def test_capacity_delay_expires_full_study_even_though_next_batch_would_fit(self):
        self.guard(now=1000)
        with self.assertRaisesRegex(TimeoutError, 'full-study forecast'):
            self.guard(now=7000)

    def test_evidence_backed_rolling_forecast_allows_completed_work_to_reduce_remainder(self):
        rolling = {**self.forecast, 'created_epoch': 7000,
                   'gpu_seconds_with_50_percent_contingency': 1000}
        atomic_json(self.root / 'main-live-forecast.json', rolling)
        self.guard(now=7000)
        (self.root / 'evidence.txt').write_text('changed evidence')
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.guard(now=7000)

    def test_changed_protocol_and_invalid_time_or_cost_fail_closed(self):
        for changed in ({'protocol_hash': 'wrong'}, {'created_epoch': 2000},
                        {'gpu_seconds_with_50_percent_contingency': -1},
                        {'api_remaining_with_50_percent_contingency_usd': float('nan')},
                        {'reserved_finalization_seconds': True},
                        {'forecast_fits_with_50_percent_contingency': False}):
            with self.subTest(changed=changed):
                (self.root / 'main-live-forecast.json').write_text(
                    json.dumps({**self.forecast, **changed}))
                with self.assertRaises(ValueError):
                    self.guard(now=1000)

    def test_current_native_usage_must_leave_room_for_whole_remainder(self):
        self.guard(now=1000, native={'conservative_elapsed_projection_usd': 280})
        with self.assertRaisesRegex(ValueError, 'spending limits'):
            self.guard(now=1000, native={'conservative_elapsed_projection_usd': 281})
        atomic_json(self.root / 'campaign.json', {'gpu_ceiling_usd': 300, 'api_ceiling_usd': 2})
        with self.assertRaisesRegex(ValueError, 'spending limits'):
            self.guard(now=1000, native={'conservative_elapsed_projection_usd': 0})

    def test_launch_stops_before_second_capacity_probe_or_any_paid_create(self):
        bundle = self.root / 'bundle'
        bundle.mkdir()
        atomic_json(bundle / 'protocol.json', self.protocol)
        atomic_json(bundle / 'batch.json', {'stage': 'main'})
        atomic_json(self.root / 'pilot-acceptance.json', self.forecast)
        atomic_json(self.root / 'native-balance.json', {'usd': 500})
        args = SimpleNamespace(campaign=self.root, bundle=bundle, minutes=15,
                               datacenter='auto', wait_for_capacity=True)
        clock = [1000]
        def advance(_):
            clock[0] = 7000
        unavailable = campaign_cli.checkpoint_preflight.CapacityUnavailable('none')
        with (
            patch.object(campaign_cli.time, 'time', side_effect=lambda: clock[0]),
            patch.object(campaign_cli.time, 'sleep', side_effect=advance),
            patch.object(campaign_cli.checkpoint_preflight, 'native_preflight',
                         side_effect=unavailable) as probe,
            patch.object(campaign_cli.checkpoint_run, 'run_batch') as paid,
        ):
            with self.assertRaisesRegex(TimeoutError, 'full-study forecast'):
                campaign_cli.launch(args)
        self.assertEqual(probe.call_count, 1)
        paid.assert_not_called()
