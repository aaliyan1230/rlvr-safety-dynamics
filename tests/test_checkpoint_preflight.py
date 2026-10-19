from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from rlvr_safety.checkpoint_study import atomic_json
from rlvr_safety.io import sha256_file
from rlvr_safety.provenance import verify_manifest

HERE = Path(__file__).resolve().parents[1] / 'infra/runpod'
sys.path.insert(0, str(HERE))
loaded = importlib.util.spec_from_file_location(
    'checkpoint_preflight', HERE / 'checkpoint_preflight.py')
guard = importlib.util.module_from_spec(loaded)
loaded.loader.exec_module(guard)


class NativePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / 'bundle'
        self.bundle.mkdir()
        self.protocol = {'gpu': guard.GPU_CHOICES[0][0], 'image': 'pinned-image',
                         'generation': {'max_new_tokens': 8192, 'do_sample': False}}
        atomic_json(self.bundle / 'protocol.json', self.protocol)
        atomic_json(self.bundle / 'batch.json', {'stage': 'canary'})
        atomic_json(self.bundle / 'bundle_manifest.json', {'files': {
            'protocol.json': sha256_file(self.bundle / 'protocol.json'),
            'batch.json': sha256_file(self.bundle / 'batch.json')}})
        self.bundle.with_suffix('.tar.gz').write_bytes(b'test prepared archive')
        (self.root / 'proof.txt').write_text('verified unpaid fixture')
        atomic_json(self.root / 'unpaid-readiness.json', {
            'stage_a': 'PASS', 'stage_b': 'PASS',
            'protocol_fingerprint': guard.readiness_fingerprint(self.protocol),
            'evidence': {'proof.txt': sha256_file(self.root / 'proof.txt')}})
        atomic_json(self.root / 'campaign.json', {
            'deadline': (datetime.now(UTC) + timedelta(hours=48)).isoformat(),
            'gpu_ceiling_usd': 300, 'api_ceiling_usd': 25})
        self.billing = {'metadata': {'totals': {'totalAmount': 29}}}
        atomic_json(self.root / 'billing-before.json', self.billing)
        self.balance = {'usd': 580, 'source': 'RunPod native billing page',
                        'observed_epoch': time.time()}
        self.launch = {'gpu': self.protocol['gpu'], 'image': 'pinned-image', 'stage': 'canary',
                       'hourly_ceiling': 4.5, 'disk_gb': 80, 'max_seconds': 5400,
                       'datacenter': 'test-dc'}
        self.catalog = {'gpus': [{'id': self.protocol['gpu'], 'price': {'secure': 3.99},
                                 'dataCenters': [{'id': 'test-dc', 'availability': 'HIGH'}]}]}
        prepared_boundary = patch.object(guard, 'verify_prepared_bundle',
            side_effect=lambda directory: verify_manifest(directory / 'bundle_manifest.json'))
        prepared_boundary.start()
        self.addCleanup(prepared_boundary.stop)

    def request(self, path, params=None):
        return {'/catalog/gpus': self.catalog, '/network-volumes': {'networkVolumes': []},
                '/billing': self.billing}[path]

    def check(self):
        return guard.native_preflight(self.bundle, self.root,
                                      balance=self.balance, launch=self.launch)

    def test_native_checks_and_maximum_exposure_pass_without_mutating_any_resources(self):
        with patch.object(guard, 'request', side_effect=self.request), \
                patch.object(guard, 'list_pods', return_value=[]):
            report = self.check()
            self.assertTrue(report['ok'])
            self.assertLess(report['maximum_launch_exposure_usd'], 10)
            self.assertEqual(report['native_usage_delta_usd'], 0)
            self.assertEqual(report['bundle_sha256'], sha256_file(
                self.bundle.with_suffix('.tar.gz')))
            self.assertEqual(guard.choose_gpu(self.catalog)['gpu'], self.protocol['gpu'])

    def test_capacity_balance_integrity_and_canary_budget_fail_closed(self):
        cases = ('capacity', 'balance', 'balance_nan', 'evidence', 'usage', 'canary_lifetime')
        for case in cases:
            with self.subTest(case=case):
                # setUp gives each failure a clean, independent campaign fixture.
                self.setUp()
                if case == 'capacity':
                    self.catalog['gpus'][0]['dataCenters'][0]['availability'] = 'NONE'
                elif case == 'balance':
                    self.balance['source'] = 'estimated from deposits'
                elif case == 'balance_nan':
                    self.balance['usd'] = float('nan')
                elif case == 'evidence':
                    (self.root / 'proof.txt').write_text('changed')
                elif case == 'usage':
                    self.billing = {'metadata': {'totals': {'totalAmount': 325}}}
                else:
                    self.launch['max_seconds'] = 21600
                with patch.object(guard, 'request', side_effect=self.request), \
                        patch.object(guard, 'list_pods', return_value=[]):
                    with self.assertRaises(ValueError):
                        self.check()

    def test_existing_or_unverified_resources_block_new_launches(self):
        with patch.object(guard, 'request', side_effect=self.request), \
                patch.object(guard, 'list_pods', return_value=[{'id': 'unexpected'}]):
            with self.assertRaisesRegex(ValueError, 'unexpected account resources'):
                self.check()
        previous = self.root / 'cloud/previous/state.json'
        atomic_json(previous, {'cleanup_verified': False})
        with patch.object(guard, 'request', side_effect=self.request), \
                patch.object(guard, 'list_pods', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'unverified cleanup'):
                self.check()

    def test_auto_placement_is_selected_in_read_only_preflight_and_never_changes_gpu(self):
        self.launch['datacenter'] = 'auto'
        with patch.object(guard, 'request', side_effect=self.request), \
                patch.object(guard, 'list_pods', return_value=[]):
            report = self.check()
            self.assertEqual(report['launch']['datacenter'], 'test-dc')
            self.assertEqual(report['launch']['gpu'], self.protocol['gpu'])
            self.catalog['gpus'][0]['dataCenters'][0]['availability'] = 'NONE'
            with self.assertRaises(guard.CapacityUnavailable):
                self.check()


if __name__ == '__main__':
    unittest.main()
