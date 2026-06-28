import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from harnessgym.artifacts import create_initial_result, ensure_harness_dirs
from harnessgym.config import RunConfig
from harnessgym.models import IterationContext, Registry
from harnessgym.runners.exec_runner import ExecRunner, parse_session_id


class ExecRunnerTests(unittest.TestCase):
    def test_exec_runner_command_construction(self) -> None:
        runner = ExecRunner(codex_bin="codex")

        self.assertEqual(
            runner.build_command("do work"),
            ["codex", "exec", "--skip-git-repo-check", "do work"],
        )
        self.assertEqual(
            runner.build_command("reflect", session_id="session-123"),
            ["codex", "exec", "--skip-git-repo-check", "resume", "session-123", "reflect"],
        )

    def test_exec_runner_injects_project_mcp_config_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            activation_dir = workspace / ".harnessgym"
            activation_dir.mkdir()
            (activation_dir / "activation.json").write_text(
                json.dumps(
                    {
                        "mcp_servers": [
                            {
                                "name": "cpu_attention",
                                "command": "python3",
                                "args": [".harnessgym/mcp/cpu_attention/server.py"],
                                "cwd": str(workspace),
                                "startup_timeout_sec": 5,
                                "tool_timeout_sec": 120,
                                "enabled_tools": ["inspect", "search"],
                                "smoke": {"status": "passed"},
                                "self_test": {"status": "passed"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            command = ExecRunner(codex_bin="codex").build_command("do work", workspace=workspace)

            self.assertEqual(command[0:3], ["codex", "exec", "--skip-git-repo-check"])
            self.assertEqual(command[-1], "do work")
            self.assertIn(f"mcp_servers.cpu_attention.command={json.dumps(sys.executable)}", command)
            args_override = next(
                part for part in command if part.startswith("mcp_servers.cpu_attention.args=")
            )
            self.assertIn("mcp_telemetry_proxy.py", args_override)
            self.assertIn("--server-name", args_override)
            self.assertIn("cpu_attention", args_override)
            self.assertIn("--response-timeout", args_override)
            self.assertIn("120", args_override)
            self.assertIn(".harnessgym/mcp/cpu_attention/server.py", args_override)
            self.assertIn(f'mcp_servers.cpu_attention.cwd="{workspace}"', command)
            self.assertIn("mcp_servers.cpu_attention.tool_timeout_sec=125", command)
            self.assertIn('mcp_servers.cpu_attention.enabled_tools=["inspect", "search"]', command)

    def test_exec_runner_skips_failed_project_mcp_self_tests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            activation_dir = workspace / ".harnessgym"
            activation_dir.mkdir()
            (activation_dir / "activation.json").write_text(
                json.dumps(
                    {
                        "mcp_servers": [
                            {
                                "name": "broken",
                                "command": "python3",
                                "args": ["server.py"],
                                "cwd": str(workspace),
                                "smoke": {"status": "passed"},
                                "self_test": {"status": "failed"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            command = ExecRunner(codex_bin="codex").build_command("do work", workspace=workspace)

            self.assertEqual(command, ["codex", "exec", "--skip-git-repo-check", "do work"])

    def test_parse_session_id_from_common_output(self) -> None:
        self.assertEqual(parse_session_id("Session ID: abc-123"), "abc-123")
        self.assertEqual(parse_session_id("session_id=xyz.789"), "xyz.789")
        self.assertIsNone(parse_session_id("no session here"))

    def test_reflection_requires_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            harness_dir = ensure_harness_dirs(workspace)
            iteration_dir = workspace / ".harnessgym" / "runs" / "run" / "iterations" / "1"
            iteration_dir.mkdir(parents=True)
            result_path = iteration_dir / "result.json"
            registry = Registry()
            create_initial_result(
                path=result_path,
                run_id="run",
                iteration=1,
                task_path=None,
                registry=registry,
            )
            config = RunConfig.from_values(workspace=workspace, task_text="Task")
            context = IterationContext(
                run_id="run",
                iteration=1,
                workspace=workspace,
                harness_dir=harness_dir,
                run_dir=iteration_dir.parents[1],
                iteration_dir=iteration_dir,
                result_path=result_path,
                registry=registry,
                task_text="Task",
            )

            result = ExecRunner(codex_bin="codex").reflect(config, context, "reflect", session_id=None)

            self.assertEqual(result.status, "failed")
            self.assertIn("no Codex session id", result.message)

    def test_timeout_does_not_hang_on_child_process_holding_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            fake_codex = workspace / "fake_codex.py"
            fake_codex.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import subprocess",
                        "import sys",
                        "import time",
                        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])",
                        "time.sleep(30)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)
            harness_dir = ensure_harness_dirs(workspace)
            iteration_dir = workspace / ".harnessgym" / "runs" / "run" / "iterations" / "1"
            iteration_dir.mkdir(parents=True)
            result_path = iteration_dir / "result.json"
            registry = Registry()
            create_initial_result(
                path=result_path,
                run_id="run",
                iteration=1,
                task_path=None,
                registry=registry,
            )
            config = RunConfig.from_values(
                workspace=workspace,
                task_text="Task",
                attempt_timeout="1s",
                codex_bin=str(fake_codex),
            )
            context = IterationContext(
                run_id="run",
                iteration=1,
                workspace=workspace,
                harness_dir=harness_dir,
                run_dir=iteration_dir.parents[1],
                iteration_dir=iteration_dir,
                result_path=result_path,
                registry=registry,
                task_text="Task",
            )

            started = time.monotonic()
            result = ExecRunner(codex_bin=str(fake_codex)).start_attempt(config, context, "attempt")
            duration = time.monotonic() - started

            self.assertEqual(result.status, "timeout")
            self.assertLess(duration, 10)
