from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / 'infra/runpod'
spec = importlib.util.spec_from_file_location('runpod_pilot', HERE / 'pilot.py')
pilot = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules):
    sys.path.insert(0, str(HERE))
    try:
        spec.loader.exec_module(pilot)
        cli = sys.modules['runpod_cli']
    finally:
        sys.path.remove(str(HERE))



class RunPodPilotTests(unittest.TestCase):
    def test_cleanup_preserves_unrelated_resources(self):
        state = {'name': 'unique-run', 'pod_ids': ['known'], 'volume_id': 'owned-volume'}
        with patch.object(pilot, 'list_pods', return_value=[
            {'id': 'recovered', 'name': 'unique-run-1'},
            {'id': 'unrelated', 'name': 'another-run-1'},
        ]), patch.object(pilot, 'terminate') as terminate, patch.object(
            pilot, 'request', side_effect=[
                {'networkVolumes': [
                    {'id': 'owned-volume', 'name': 'unique-run'},
                    {'id': 'unrelated-volume', 'name': 'another-run'},
                ]}, {},
            ],
        ) as request, patch.object(pilot, 'deleted', return_value=True):
            pilot.cleanup(state)
        assert {c.args[0] for c in terminate.call_args_list} == {'known', 'recovered'}
        assert request.call_args.args == ('/network-volumes/owned-volume',)
        assert request.call_args.kwargs == {'method': 'DELETE'}


    def test_unconfirmed_deletion_fails(self):
        with patch.object(pilot, 'request', return_value={}), patch.object(
            pilot, 'deleted', return_value=False
        ), patch.object(pilot.time, 'sleep'), self.assertRaisesRegex(RuntimeError, 'not verified'):
            pilot.terminate('pod')


    def test_cleanup_runs_when_preflight_fails(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(yes=True, output=Path(directory) / 'run', datacenter='EUR-IS-1')
            with patch.object(pilot.subprocess, 'Popen', return_value=SimpleNamespace(pid=123)), \
                 patch.object(pilot, 'request', side_effect=pilot.ApiError('catalog offline')), \
                 patch.object(pilot, 'cleanup') as cleanup, \
                 self.assertRaises(pilot.ApiError):
                pilot.run(args)
            cleanup.assert_called_once()
            assert json.loads((args.output / 'state.json').read_text())['cleanup_verified']


    def test_accepted_price_ceiling_preserves_id_before_failure(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'state.json'
            state = {'name': 'run', 'pod_ids': [], 'volume_id': 'volume'}
            args = SimpleNamespace(datacenter='EUR-IS-1')
            with patch.object(pilot, 'request', return_value={'id': 'paid-pod', 'cost': 99}), \
                 self.assertRaisesRegex(RuntimeError, 'price'):
                pilot.create_pod(args, state, path, 1)
            assert json.loads(path.read_text())['pod_ids'] == ['paid-pod']


    def test_post_and_empty_delete_response_keep_secrets_off_argv(self):
        result = subprocess.CompletedProcess([], 0, '\n' + cli.MARKER + '204', '')
        with patch.object(cli, 'api_key', return_value='private-key'), patch.object(
            cli.subprocess, 'run', return_value=result
        ) as run:
            assert cli.request('/pods/id', method='DELETE') == {}
            self.assertEqual(cli.request(
                '/pods', method='POST', body={'env': {'SECRET': 'private-env'}}), {})
        assert 'private-key' not in str(run.call_args.args)
        assert 'private-env' not in str(run.call_args.args)
        assert 'private-env' in run.call_args.kwargs['input']
