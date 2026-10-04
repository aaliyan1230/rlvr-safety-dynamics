from __future__ import annotations

import importlib.util
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

CLI_PATH = Path(__file__).resolve().parents[1] / "infra/runpod/runpod_cli.py"
spec = importlib.util.spec_from_file_location("runpod_cli", CLI_PATH)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class RunPodCliTests(unittest.TestCase):
    def test_owner_only_env_and_no_shell_execution(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text("OTHER=secret\nRUNPOD_API_KEY='$(false);literal'\n")
            path.chmod(0o600)
            self.assertEqual(cli.api_key(path), "$(false);literal")
            self.assertNotIn("OTHER", os.environ)
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "owner access"):
                cli.api_key(path)

    def test_exported_key_precedence(self):
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "exported"}):
            self.assertEqual(cli.api_key(Path("/nonexistent/env")), "exported")

    def test_credentials_stdin_header_and_query_encoding(self):
        result = subprocess.CompletedProcess([], 0, '{"pods":[]}\n' + cli.MARKER + "200", "")
        with patch.object(cli, "api_key", return_value="test-key"), patch.object(
            cli.subprocess, "run", return_value=result
        ) as run, patch.dict(os.environ, {"RUNPOD_API_KEY": "test-key"}):
            self.assertEqual(cli.request("/pods", {"cursor": "a&b"}), {"pods": []})
        args, kwargs = run.call_args
        self.assertNotIn("test-key", repr(args))
        self.assertIn("Authorization: Bearer test-key", kwargs["input"])
        self.assertNotIn("RUNPOD_API_KEY", kwargs["env"])
        self.assertIn("User-Agent: " + cli.USER_AGENT, args[0])
        self.assertEqual(args[0][-1], cli.API_BASE + "/pods?cursor=a%26b")

    def test_errors_do_not_echo_secrets(self):
        result = subprocess.CompletedProcess([], 0, 'secret\n' + cli.MARKER + "403", "")
        with patch.object(cli, "api_key", return_value="secret"), patch.object(
            cli.subprocess, "run", return_value=result
        ):
            with self.assertRaises(cli.ApiError) as caught:
                cli.request("/pods")
            self.assertEqual(caught.exception.status, 403)
            self.assertNotIn("secret", str(caught.exception))

    def test_inventory_pagination_and_malformed_response(self):
        with patch.object(cli, "request", side_effect=[
            {"pods": [{"id": "one"}], "pagination": {"hasNextPage": True, "nextCursor": "x"}},
            {"pods": [{"id": "two"}], "pagination": {"hasNextPage": False}},
        ]) as request:
            self.assertEqual([p["id"] for p in cli.list_pods()], ["one", "two"])
            self.assertEqual(request.call_args.args, ("/pods", {"cursor": "x"}))
        with patch.object(cli, "request", return_value={}):
            with self.assertRaises(cli.ApiError):
                cli.list_pods()

    def test_repeated_cursor_fails_instead_of_hiding_resources(self):
        response = {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "x"}}
        with patch.object(cli, "request", return_value=response):
            with self.assertRaisesRegex(cli.ApiError, "pagination"):
                cli.list_pods()

    def test_pod_output_omits_sensitive_configuration(self):
        self.assertEqual(cli.public_pod({
            "id": "pod", "env": {"API_KEY": "private"}, "args": "private-command",
            "registry": "private-registry", "image": "private-image",
        }), {"id": "pod"})


if __name__ == "__main__":
    unittest.main()
