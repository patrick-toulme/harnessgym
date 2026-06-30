import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from harnessgym.artifacts import create_initial_result, ensure_harness_dirs
from harnessgym.config import RunConfig
from harnessgym.models import IterationContext, Registry
from harnessgym.runners.claude_runner import (
    ClaudeRunner,
    parse_claude_result,
    parse_claude_session_id,
)


class ClaudeRunnerTests(unittest.TestCase):
    def test_claude_runner_command_construction(self) -> None:
        runner = ClaudeRunner(claude_bin="claude", model="sonnet", max_budget_usd=1.5)

        command = runner.build_command("do work")

        self.assertEqual(command[0:3], ["claude", "-p", "--output-format"])
        self.assertIn("json", command)
        self.assertIn("--permission-mode", command)
        self.assertIn("bypassPermissions", command)
        self.assertIn("--model", command)
        self.assertIn("sonnet", command)
        self.assertIn("--max-budget-usd", command)
        self.assertIn("1.5", command)
        self.assertEqual(command[-1], "do work")

    def test_claude_runner_resume_command(self) -> None:
        command = ClaudeRunner(claude_bin="claude").build_command(
            "reflect",
            session_id="44b64ca5-735d-4d10-8270-4aef483c00a1",
        )

        self.assertIn("--resume", command)
        self.assertIn("44b64ca5-735d-4d10-8270-4aef483c00a1", command)
        self.assertEqual(command[-1], "reflect")

    def test_claude_runner_writes_mcp_config_and_allowed_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            activation_dir = workspace / ".harnessgym"
            activation_dir.mkdir()
            server_path = workspace / ".harnessgym" / "mcp" / "tensor-plan" / "server.py"
            server_path.parent.mkdir(parents=True)
            server_path.write_text("print('server')\n", encoding="utf-8")
            (activation_dir / "activation.json").write_text(
                json.dumps(
                    {
                        "mcp_servers": [
                            {
                                "name": "tensor-plan",
                                "command": "python3",
                                "args": [".harnessgym/mcp/tensor-plan/server.py"],
                                "cwd": str(workspace),
                                "enabled_tools": ["benchmark_plan", "apply_candidate"],
                                "tool_timeout_sec": 180,
                                "smoke": {"status": "passed"},
                                "self_test": {"status": "passed"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            command = ClaudeRunner(claude_bin="claude").build_command("do work", workspace=workspace)

            self.assertEqual(command[-1], "do work")
            self.assertIn("--strict-mcp-config", command)
            self.assertIn("--mcp-config", command)
            self.assertIn("--allowedTools=mcp__tensor-plan", command)
            config_path = workspace / ".harnessgym" / "claude_mcp_config.json"
            self.assertIn(str(config_path), command)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["mcpServers"]["tensor-plan"]["command"], sys.executable)
            args = config["mcpServers"]["tensor-plan"]["args"]
            self.assertTrue(args[0].endswith("claude_mcp_bridge.py"))
            self.assertEqual(
                args[1:8],
                ["--server-name", "tensor-plan", "--response-timeout", "180", "--cwd", str(workspace), "--"],
            )
            self.assertEqual(args[8:], ["python3", str((workspace / ".harnessgym/mcp/tensor-plan/server.py").resolve())])
            self.assertEqual(config["mcpServers"]["tensor-plan"]["type"], "stdio")
            self.assertEqual(config["mcpServers"]["tensor-plan"]["env"]["HARNESSGYM_WORKSPACE"], str(workspace))

    def test_claude_runner_skips_failed_mcp_self_tests(self) -> None:
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

            command = ClaudeRunner(claude_bin="claude").build_command("do work", workspace=workspace)

            self.assertNotIn("--mcp-config", command)
            self.assertFalse((workspace / ".harnessgym" / "claude_mcp_config.json").exists())

    def test_parse_claude_session_id_from_json(self) -> None:
        payload = {
            "type": "result",
            "subtype": "success",
            "result": "OK",
            "session_id": "44b64ca5-735d-4d10-8270-4aef483c00a1",
        }

        text = json.dumps(payload)

        self.assertEqual(parse_claude_result(text), payload)
        self.assertEqual(parse_claude_session_id(text), "44b64ca5-735d-4d10-8270-4aef483c00a1")

    def test_parse_claude_result_prioritizes_result_over_system_in_jsonl(self) -> None:
        # JSONL with [assistant, result, system] — reversed scan must pick
        # the result message, not the trailing system/init message that also
        # carries session_id.
        jsonl = "\n".join(
            [
                json.dumps({"type": "assistant", "message": "working"}),
                json.dumps(
                    {
                        "type": "result",
                        "session_id": "abc-123",
                        "result": "TASK SOLVED",
                        "is_error": False,
                    }
                ),
                json.dumps({"type": "system", "subtype": "init", "session_id": "xyz-456"}),
            ]
        )

        parsed = parse_claude_result(jsonl)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["type"], "result")
        self.assertEqual(parsed["session_id"], "abc-123")
        self.assertEqual(parsed["result"], "TASK SOLVED")

    def test_parse_claude_result_falls_back_to_session_id_when_no_result_type(self) -> None:
        # When no type=="result" line exists, fall back to session_id match.
        jsonl = "\n".join(
            [
                json.dumps({"type": "assistant", "message": "working"}),
                json.dumps({"type": "system", "subtype": "init", "session_id": "xyz-456"}),
            ]
        )

        parsed = parse_claude_result(jsonl)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["session_id"], "xyz-456")

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
            config = RunConfig.from_values(workspace=workspace, task_text="Task", runner="claude")
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

            result = ClaudeRunner(claude_bin="claude").reflect(config, context, "reflect", session_id=None)

            self.assertEqual(result.status, "failed")
            self.assertIn("no Claude Code session id", result.message)

    def test_timeout_does_not_hang_on_child_process_holding_pipes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            fake_claude = workspace / "fake_claude.py"
            fake_claude.write_text(
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
            fake_claude.chmod(0o755)
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
                runner="claude",
                attempt_timeout="1s",
                claude_bin=str(fake_claude),
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
            result = ClaudeRunner(claude_bin=str(fake_claude)).start_attempt(config, context, "attempt")
            duration = time.monotonic() - started

            self.assertEqual(result.status, "timeout")
            self.assertLess(duration, 10)

    def test_transcript_records_claude_api_error_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            fake_claude = workspace / "fake_claude.py"
            fake_claude.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import json",
                        "import sys",
                        "payload = {",
                        "    'type': 'result',",
                        "    'subtype': 'success',",
                        "    'is_error': True,",
                        "    'api_error_status': 429,",
                        "    'result': 'quota blocked',",
                        "    'session_id': '44b64ca5-735d-4d10-8270-4aef483c00a1',",
                        "}",
                        "sys.stdout.write(json.dumps(payload))",
                        "sys.exit(1)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            fake_claude.chmod(0o755)
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
                runner="claude",
                claude_bin=str(fake_claude),
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

            result = ClaudeRunner(claude_bin=str(fake_claude)).start_attempt(config, context, "attempt")

            self.assertEqual(result.status, "failed")
            transcript = Path(result.transcript_path or "").read_text(encoding="utf-8")
            self.assertIn("claude_api_error_status: 429", transcript)
            self.assertIn("session_id: 44b64ca5-735d-4d10-8270-4aef483c00a1", transcript)
