from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rlvr_safety.checkpoint_study import ResponseStore, build_matrix, digest, seal_chunk
from rlvr_safety.io import write_jsonl

HERE = Path(__file__).resolve().parents[1] / 'infra/runpod'
sys.path.insert(0, str(HERE))
loaded = importlib.util.spec_from_file_location('checkpoint_run', HERE / 'checkpoint_run.py')
runner = importlib.util.module_from_spec(loaded)
loaded.loader.exec_module(runner)


class CheckpointCloudTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bundle = self.root / 'bundle'
        bundle.mkdir()
        (self.root / 'bundle.tar.gz').write_bytes(b'test archive')
        key = self.root / 'key'
        key.write_text('not-a-real-private-key')
        key.with_suffix('.pub').write_text('test-public-key')
        self.args = SimpleNamespace(bundle=bundle, private_key=key,
                                    output=self.root / 'run', deadline=time.time() + 4000)
        self.report = {'ok': True, 'checked_epoch': time.time(),
                       'bundle_sha256': runner.sha256_file(self.root / 'bundle.tar.gz'), 'launch': {
            'max_seconds': 1800, 'image': 'pinned-image', 'gpu': 'A100',
            'datacenter': 'US-KS-2', 'disk_gb': 80, 'hourly_ceiling': 2}}
        self.creates = []
        self.pod = {'id': 'test-pod', 'cost': 1.79, 'status': 'RUNNING',
                    'ssh': {'direct': {'host': 'host', 'port': 22}}}

    def api(self, path, params=None, *, method='GET', body=None):
        if path == '/account/ssh-keys':
            return {'keys': ['test-public-key']}
        if path == '/pods' and method == 'POST':
            self.creates.append(body)
            return self.pod
        if path == '/pods/test-pod':
            saved = json.loads((self.args.output / 'state.json').read_text())
            self.assertEqual(saved['pod_ids'], ['test-pod'])
            return self.pod
        if path == '/network-volumes':
            return {'networkVolumes': []}
        raise AssertionError((path, method))

    def ssh(self, connection, args, command, **kwargs):
        output = b'0\n' if 'exit.code' in command else b'123\n'
        if command.startswith('sha256sum'):
            output = (runner.sha256_file(self.root / 'bundle.tar.gz') + ' bundle').encode()
        return SimpleNamespace(stdout=output, returncode=0)

    def retrieve(self, *args):
        path = self.args.output / 'retrieved'
        path.mkdir(exist_ok=True)
        (path / 'run_summary.json').write_text('{"completed":true}')
        return 2

    def patches(self, stack):
        mocks = {}
        for name, kwargs in {
            'request': {'side_effect': self.api}, 'list_pods': {'return_value': []},
            'retrieve': {'side_effect': self.retrieve},
            'verify_retrieved': {'return_value': 24},
        }.items():
            mocks[name] = stack.enter_context(patch.object(runner, name, **kwargs))
        stack.enter_context(patch.object(runner.permission_run, 'ssh', side_effect=self.ssh))
        stack.enter_context(patch.object(runner, 'verify_manifest'))
        stack.enter_context(patch.object(runner.permission_run, 'verify_archive'))
        mocks['cleanup'] = stack.enter_context(patch.object(runner.pilot, 'cleanup'))
        stack.enter_context(patch.object(runner.subprocess, 'Popen',
                                         return_value=SimpleNamespace(pid=1)))
        return mocks

    def test_success_verifies_retrieved_data_and_backup_before_passing_then_deletes(self):
        with ExitStack() as stack:
            mocks = self.patches(stack)
            backup = Mock()
            state = runner.run_batch(self.args, self.report, backup=backup)
            self.assertTrue(state['passed'] and state['backup_verified']
                            and state['cleanup_verified'])
            mocks['verify_retrieved'].assert_called_once()
            mocks['cleanup'].assert_called_once()
            backup.assert_called_once()
            self.assertEqual(self.creates[0]['gpu']['count'], 1)
            self.assertNotIn('mounts', self.creates[0])

    def test_real_retrieved_validation_requires_every_expected_answer_and_mapping(self):
        questions = build_matrix(HERE.parents[1])[:2]
        protocol = {'test': 'transport'}
        write_jsonl(self.args.bundle / 'matrix.jsonl', questions)
        (self.args.bundle / 'protocol.json').write_text(json.dumps(protocol))
        directory = self.args.output / 'retrieved' / 'think_0000'
        store = ResponseStore(directory, protocol, questions)

        def output(row):
            return {'id': row['id'], 'input_hash': row['input_hash'],
                    'protocol_hash': digest(protocol), 'model_revision_resolved': 'a' * 40,
                    'raw_response': 'CHOICE: A', 'reasoning_open': False,
                    'generated_tokens': 5, 'stop': 'complete'}

        store.append(output(questions[0]))
        first = seal_chunk(store)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            runner.verify_retrieved(self.args)
        store.append(output(questions[1]))
        seal_chunk(store)
        self.assertEqual(runner.verify_retrieved(self.args), 2)
        (first / 'responses.jsonl').write_text('corrupted evidence')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            runner.verify_retrieved(self.args)

    def test_disconnect_bad_retrieval_or_backup_failure_still_deletes(self):
        for fault in ('disconnect', 'retrieval', 'backup', 'validation'):
            with self.subTest(fault=fault), ExitStack() as stack:
                self.args.output = self.root / fault
                mocks = self.patches(stack)
                backup = Mock()
                if fault == 'disconnect':
                    stack.enter_context(patch.object(runner.permission_run, 'ssh',
                                                     side_effect=TimeoutError('disconnect')))
                elif fault == 'retrieval':
                    mocks['retrieve'].side_effect = ValueError('corrupt retrieval')
                elif fault == 'validation':
                    mocks['verify_retrieved'].side_effect = ValueError('missing answers')
                else:
                    backup.side_effect = RuntimeError('private backup failed')
                with self.assertRaises((TimeoutError, ValueError, RuntimeError)):
                    runner.run_batch(self.args, self.report, backup=backup)
                mocks['cleanup'].assert_called_once()
                state = json.loads((self.args.output / 'state.json').read_text())
                self.assertFalse(state['passed'])
                self.assertTrue(state['cleanup_verified'])

    def test_stale_failed_or_expired_preflight_never_creates_paid_resource(self):
        with ExitStack() as stack:
            mocks = self.patches(stack)
            for changes in ({'ok': False}, {'checked_epoch': time.time() - 1000}):
                with self.assertRaises(ValueError):
                    runner.run_batch(self.args, {**self.report, **changes}, backup=Mock())
            self.args.deadline = time.time() + 10
            with self.assertRaises(ValueError):
                runner.run_batch(self.args, self.report, backup=Mock())
            mocks['request'].assert_not_called()

    def test_cleanup_failure_never_leaves_a_passing_state(self):
        with ExitStack() as stack:
            mocks = self.patches(stack)
            mocks['cleanup'].side_effect = RuntimeError('deletion not verified')
            with self.assertRaisesRegex(RuntimeError, 'deletion not verified'):
                runner.run_batch(self.args, self.report, backup=Mock())
            saved = json.loads((self.args.output / 'state.json').read_text())
            self.assertFalse(saved['passed'])
            self.assertFalse(saved['cleanup_verified'])

    def test_quote_rejection_and_ambiguous_create_record_identity_before_cleanup(self):
        for fault in ('quote', 'ambiguous'):
            with self.subTest(fault=fault), ExitStack() as stack:
                self.args.output = self.root / fault
                mocks = self.patches(stack)
                if fault == 'quote':
                    self.pod['cost'] = 20
                else:
                    def ambiguous(path, *args, **kwargs):
                        if path == '/pods':
                            self.creates.append(kwargs['body'])
                            raise TimeoutError('ambiguous create')
                        return self.api(path, *args, **kwargs)
                    mocks['request'].side_effect = ambiguous
                with self.assertRaises((RuntimeError, TimeoutError)):
                    runner.run_batch(self.args, self.report, backup=Mock())
                saved = json.loads((self.args.output / 'state.json').read_text())
                self.assertTrue(saved['name'].startswith('rlvr-olmo-'))
                self.assertEqual(len(self.creates), 1 if fault == 'quote' else 2)
                mocks['cleanup'].assert_called_once()


if __name__ == '__main__':
    unittest.main()
