from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path

from rlvr_safety.checkpoint_bundle import build_bundle
from rlvr_safety.checkpoint_protocol import select_questions
from rlvr_safety.checkpoint_study import atomic_json, build_matrix, digest, verify_chunk
from rlvr_safety.io import write_jsonl
from rlvr_safety.provenance import verify_manifest

ROOT = Path(__file__).resolve().parents[1]


class CheckpointBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.prepared = self.directory / 'prepared'
        (self.prepared / 'tokenizer').mkdir(parents=True)
        (self.prepared / 'tokenizer/chat_template.jinja').write_text('fixture <think>')
        self.matrix = build_matrix(ROOT)
        for row in self.matrix:
            row.update(revision=f"{row['step']:040x}", repo='test/model')
            row['input_hash'] = digest({k: v for k, v in row.items() if k != 'input_hash'})
        write_jsonl(self.prepared / 'matrix.jsonl', self.matrix)
        atomic_json(self.prepared / 'model_metadata.json', [
            {'step': step, 'revision': f'{step:040x}', 'repo': 'test/model', 'siblings': []}
            for step in sorted({r['step'] for r in self.matrix})])

    def bundle(self, stage='canary', **changes):
        destination = self.directory / 'bundle'
        build_bundle(ROOT, self.prepared, destination, stage=stage,
                     image='runpod/pytorch@sha256:' + '0' * 64, gpu='test-GPU',
                     campaign_deadline=time.time() + 600, **changes)
        return destination

    def run_worker(self, bundle, *extra):
        return subprocess.run([sys.executable, str(bundle / 'checkpoint_workload.py'),
                               '--bundle', str(bundle), '--simulate', *extra],
                              env={**os.environ, 'PYTHONPATH': str(bundle / 'src')},
                              capture_output=True, text=True, timeout=60)

    def test_selections_preserve_counts_coverage_and_deduplicate_pilot(self):
        canary = select_questions(self.matrix, 'canary')
        self.assertEqual(len(canary), 24)
        self.assertEqual(sum(bool(r.get('engineering_control')) for r in canary), 12)
        for step in (0, 1375):
            cells = [r for r in canary if r['step'] == step and not r.get('engineering_control')]
            self.assertEqual(sorted(Counter(r['category'] for r in cells).values()), [2, 2, 2])
            self.assertEqual(Counter(r['panel'] for r in cells), {'mcq': 3, 'written': 3})
        pilot = select_questions(self.matrix, 'pilot')
        self.assertEqual(len(pilot), len({r['id'] for r in pilot}))
        self.assertEqual(sum(r['panel'] == 'written' for r in pilot), 12)
        main = select_questions(self.matrix, 'main')
        self.assertEqual(len(main), 7056)
        first = [r for r in main if r['step'] == 0][:24]
        self.assertEqual(len({r['source_id'] for r in first}), 24)
        self.assertEqual(len({r['wording_id'] for r in first}), 4)
        self.assertEqual(len({r['option_order'] for r in first}), 6)

    def test_actual_prepared_entrypoint_seals_interrupted_tail_and_resumes(self):
        bundle = self.bundle()
        failed = self.run_worker(bundle, '--interrupt-after', '5')
        self.assertNotEqual(failed.returncode, 0)
        chunks = list((bundle / 'out').glob('*/chunks/*/manifest.json'))
        self.assertEqual(sum(len(verify_chunk(p.parent)) for p in chunks), 5)
        finished = self.run_worker(bundle)
        self.assertEqual(finished.returncode, 0, finished.stderr)
        summary = json.loads((bundle / 'out/run_summary.json').read_text())
        self.assertTrue(summary['completed'] and summary['simulation'])
        self.assertEqual(summary['records'], 24)
        rows = [r for p in (bundle / 'out').glob('*/chunks/*/manifest.json')
                for r in verify_chunk(p.parent)]
        self.assertEqual(len(rows), len({r['id'] for r in rows}))
        self.assertTrue(all(r['simulation'] for r in rows))
        verify_manifest(bundle / 'bundle_manifest.json')

    def test_changed_prepared_input_is_rejected_before_any_model_or_provider_call(self):
        bundle = self.bundle()
        (bundle / 'matrix.jsonl').write_text('{}\n')
        result = self.run_worker(bundle)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum mismatch', result.stderr)
        self.assertFalse((bundle / 'out').exists())

    def test_real_bootstrap_requires_hashed_dependencies_and_no_simulation(self):
        bundle = self.bundle()
        script = (bundle / 'bootstrap.sh').read_text()
        self.assertIn('--require-hashes', script)
        self.assertIn('python3.12 -m venv', script)
        self.assertIn('--no-deps', script)
        self.assertNotIn('--simulate', script)
        protocol = json.loads((bundle / 'protocol.json').read_text())
        self.assertGreater(len(protocol['runtime_pins']), 30)
        self.assertEqual(protocol['generation']['max_new_tokens'], 8192)
        self.assertNotIn('api_key', protocol)


if __name__ == '__main__':
    unittest.main()
