from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import patch

LAMBDA_DIR = Path(__file__).resolve().parents[1] / "infra/lambda/lambda"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, LAMBDA_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


project_env = load_module("project_env")
with patch.dict(sys.modules, {"project_env": project_env}):
    lambda_cli = load_module("lambda_cli")
with patch.dict(sys.modules, {"project_env": project_env, "lambda_cli": lambda_cli}):
    run_remote = load_module("run_remote")


class LambdaEnvironmentTests(unittest.TestCase):
    def test_loads_only_lambda_values_without_executing_shell(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / ".env"
            path.write_text(
                "OTHER_KEY=private\nexport LAMBDA_API_KEY='$(false); literal'\n"
                "LAMBDA_FORWARD_API_KEY=1 # opted in\n"
            )
            path.chmod(0o600)
            project_env.load_project_env(path)
            self.assertEqual(os.environ["LAMBDA_API_KEY"], "$(false); literal")
            self.assertEqual(os.environ["LAMBDA_FORWARD_API_KEY"], "1")
            self.assertNotIn("OTHER_KEY", os.environ)

    def test_rejects_environment_readable_by_other_users(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("LAMBDA_API_KEY=fake\n")
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "chmod 600"):
                project_env.load_project_env(path)

    def test_exported_environment_takes_precedence(self):
        with TemporaryDirectory() as directory, patch.dict(
            os.environ, {"LAMBDA_API_KEY": "exported"}, clear=True
        ):
            path = Path(directory) / ".env"
            path.write_text("LAMBDA_API_KEY=local\n")
            path.chmod(0o600)
            project_env.load_project_env(path)
            self.assertEqual(os.environ["LAMBDA_API_KEY"], "exported")

    def test_forwarding_is_opt_in_and_can_be_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(run_remote.forwarding_enabled(None))
            os.environ["LAMBDA_FORWARD_API_KEY"] = "1"
            self.assertTrue(run_remote.forwarding_enabled(None))
            self.assertFalse(run_remote.forwarding_enabled(False))
            os.environ["LAMBDA_FORWARD_API_KEY"] = "typo"
            with self.assertRaises(ValueError):
                run_remote.forwarding_enabled(None)

    def test_key_uses_stdin_only_and_command_preserves_quoting(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / "activation with ' quote.env"
            env_file.write_text("export TEST_ACTIVATION=active\nset -x\n")
            key = "fake-'quoted;$key"
            command = 'printf "%s/%s" "$LAMBDA_API_KEY" "$TEST_ACTIVATION"'
            with patch.dict(os.environ, {"LAMBDA_API_KEY": key}), patch.object(
                run_remote.subprocess, "run"
            ) as mock_run:
                mock_run.return_value.returncode = 7
                self.assertEqual(
                    run_remote.run_remote(["host"], str(env_file), directory, command, key), 7
                )
            args, kwargs = mock_run.call_args
            self.assertNotIn(key, repr(args))
            self.assertNotIn("LAMBDA_API_KEY", kwargs["env"])
            result = subprocess.run(
                ["bash", "-s"], input=kwargs["input"], text=True, capture_output=True,
                env={"PATH": os.environ["PATH"]}, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, key + "/active")
            self.assertNotIn(key, result.stderr)

    def test_disabled_forwarding_unsets_any_persistent_key(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / "activation.env"
            env_file.write_text("export LAMBDA_API_KEY=stale\n")
            with patch.object(run_remote.subprocess, "run") as mock_run:
                run_remote.run_remote(
                    ["host"], str(env_file), directory, 'printf "%s" "${LAMBDA_API_KEY-unset}"',
                    None
                )
            result = subprocess.run(
                ["bash", "-s"], input=mock_run.call_args.kwargs["input"],
                text=True, capture_output=True, check=False
            )
            self.assertEqual(result.stdout, "unset")

    def test_api_credentials_and_json_body_stay_off_curl_arguments(self):
        completed = subprocess.CompletedProcess(
            [], 0, '{"data": []}\n' + lambda_cli._STATUS_MARKER + "200", ""
        )
        with patch.object(lambda_cli, "api_key", return_value="fake-secret"), patch.object(
            lambda_cli.subprocess, "run", return_value=completed
        ) as mock_run:
            self.assertEqual(lambda_cli.request("POST", "/example", {"value": 'a"b\nc'}),
                             {"data": []})
        args, kwargs = mock_run.call_args
        self.assertNotIn("fake-secret", repr(args))
        self.assertIn("User-Agent: OpenAI File Downloader, XaiImageApiFetch/1.0", args[0])
        self.assertIn("fake-secret", kwargs["input"])
        self.assertIn("data-binary = ", kwargs["input"])

    @unittest.skipUnless(shutil.which("curl"), "curl is required by the Lambda CLI")
    def test_curl_config_preserves_auth_header_and_complex_json(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received["body"] = self.rfile.read(int(self.headers["Content-Length"]))
                received["agent"] = self.headers["User-Agent"]
                received["auth"] = self.headers["Authorization"]
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data": []}')

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body = {"value": 'quotes " and \\ and \n and \t and café'}
        try:
            with patch.object(lambda_cli, "api_key", return_value="fake-secret"), patch.object(
                lambda_cli, "API_BASE", f"http://127.0.0.1:{server.server_port}"
            ):
                self.assertEqual(lambda_cli.request("POST", "/example", body), {"data": []})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(json.loads(received["body"]), body)
        self.assertEqual(received["agent"], lambda_cli.USER_AGENT)
        self.assertEqual(received["auth"], "Basic ZmFrZS1zZWNyZXQ6")


if __name__ == "__main__":
    unittest.main()
