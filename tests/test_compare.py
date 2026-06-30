from __future__ import annotations

import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from harnessgym.compare import CompareConfig, copy_harness_artifacts, copy_workspace_template, run_compare


class CompareTests(unittest.TestCase):
    def test_copy_workspace_template_excludes_generated_state_and_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template"
            template.mkdir()
            (template / "task.md").write_text("Task.\n", encoding="utf-8")
            (template / ".harnessgym").mkdir()
            (template / ".harnessgym" / "registry.json").write_text("{}", encoding="utf-8")
            (template / ".codex").mkdir()
            (template / ".codex" / "config.toml").write_text("", encoding="utf-8")
            (template / ".agents").mkdir()
            (template / ".agents" / "skills").mkdir()
            (template / ".claude").mkdir()
            (template / ".claude" / "skills").mkdir()
            (template / "extension.so").write_text("native extension", encoding="utf-8")
            (template / "libfixture.dylib").write_text("native fixture", encoding="utf-8")
            (template / "flash_bench_final_128x64").write_text("binary-ish", encoding="utf-8")
            (template / "__pycache__").mkdir()
            (template / "__pycache__" / "x.pyc").write_text("", encoding="utf-8")

            copied = copy_workspace_template(template, root / "copy")

            self.assertEqual(copied, ["extension.so", "libfixture.dylib", "task.md"])
            self.assertTrue((root / "copy" / "task.md").exists())
            self.assertTrue((root / "copy" / "extension.so").exists())
            self.assertTrue((root / "copy" / "libfixture.dylib").exists())
            self.assertFalse((root / "copy" / ".harnessgym").exists())
            self.assertFalse((root / "copy" / ".codex").exists())
            self.assertFalse((root / "copy" / ".agents").exists())
            self.assertFalse((root / "copy" / ".claude").exists())
            self.assertFalse((root / "copy" / "flash_bench_final_128x64").exists())

    def test_copy_harness_artifacts_from_workspace_or_harness_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_workspace = root / "source"
            tool_dir = source_workspace / ".harnessgym" / "tools"
            tool_dir.mkdir(parents=True)
            (tool_dir / "probe.py").write_text("print('probe')\n", encoding="utf-8")
            tests_dir = source_workspace / ".harnessgym" / "tests"
            tests_dir.mkdir(parents=True)
            (tests_dir / "test_probe.py").write_text("def test_probe():\n    assert True\n", encoding="utf-8")
            registry = {
                "version": 1,
                "artifacts": [
                    {"kind": "tool", "path": ".harnessgym/tools/probe.py"},
                    {"kind": "test", "path": ".harnessgym/tests/test_probe.py"},
                ],
            }
            (source_workspace / ".harnessgym" / "registry.json").write_text(
                json.dumps(registry),
                encoding="utf-8",
            )
            workspace = root / "dest"
            workspace.mkdir()

            copied = copy_harness_artifacts(source_workspace, workspace)

            self.assertIn(".harnessgym/tools/probe.py", copied)
            self.assertIn(".harnessgym/tests/test_probe.py", copied)
            self.assertIn(".harnessgym/registry.json", copied)
            self.assertTrue((workspace / ".harnessgym" / "tools" / "probe.py").exists())
            self.assertTrue((workspace / ".harnessgym" / "tests" / "test_probe.py").exists())

    def test_fake_compare_report_and_artifact_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template"
            template.mkdir()
            (template / "task.md").write_text("Optimize this.\n", encoding="utf-8")
            (template / "post_score.py").write_text(
                "import json\nprint(json.dumps({'best_cycles': 7}))\n",
                encoding="utf-8",
            )
            artifact_source = root / "artifact_source"
            tool_dir = artifact_source / ".harnessgym" / "tools"
            tool_dir.mkdir(parents=True)
            (tool_dir / "probe.py").write_text("print('probe')\n", encoding="utf-8")
            output_dir = root / "compare"

            config = CompareConfig.from_values(
                workspace_template=template,
                output_dir=output_dir,
                task="task.md",
                artifact_source=artifact_source,
                runner="fake",
                trials=1,
                post_command=f"{shlex.quote(sys.executable)} post_score.py",
                post_score_key="best_cycles",
                require_active_harness=False,
            )
            report = run_compare(config)

            self.assertEqual(len(report["trials"]), 2)
            self.assertEqual(report["summary"]["plain"]["post_score"]["best"], 7.0)
            self.assertEqual(report["summary"]["harnessed"]["post_score"]["best"], 7.0)
            self.assertEqual(report["summary"]["plain"]["iterations_completed"]["best"], 1.0)
            harnessed = next(trial for trial in report["trials"] if trial["group"] == "harnessed")
            plain = next(trial for trial in report["trials"] if trial["group"] == "plain")
            self.assertIn(".harnessgym/tools/probe.py", harnessed["copied_artifacts"])
            self.assertEqual(plain["copied_artifacts"], [])
            self.assertEqual(harnessed["iterations_completed"], 1)
            self.assertEqual(len(harnessed["iterations"]), 1)
            self.assertIsNotNone(harnessed["cumulative_attempt_duration_seconds"])

            harnessed_prompt = (
                output_dir
                / "harnessed"
                / "trial_1"
                / "workspace"
                / ".harnessgym"
                / "runs"
                / "compare-harnessed-1"
                / "iterations"
                / "1"
                / "attempt.prompt.txt"
            ).read_text(encoding="utf-8")
            self.assertIn(".harnessgym/tools/probe.py", harnessed_prompt)

    def test_harnessed_compare_requires_active_mcp_tools_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template"
            template.mkdir()
            (template / "task.md").write_text("Optimize this.\n", encoding="utf-8")
            (template / "post_score.py").write_text(
                "import json\nprint(json.dumps({'best_cycles': 7}))\n",
                encoding="utf-8",
            )
            artifact_source = root / "artifact_source"
            tool_dir = artifact_source / ".harnessgym" / "tools"
            tool_dir.mkdir(parents=True)
            (tool_dir / "probe.py").write_text("print('probe')\n", encoding="utf-8")

            config = CompareConfig.from_values(
                workspace_template=template,
                output_dir=root / "compare",
                task="task.md",
                artifact_source=artifact_source,
                runner="fake",
                trials=1,
                post_command=f"{shlex.quote(sys.executable)} post_score.py",
                post_score_key="best_cycles",
            )
            report = run_compare(config)

            harnessed = next(trial for trial in report["trials"] if trial["group"] == "harnessed")
            self.assertFalse(harnessed["comparison_valid"])
            self.assertFalse(harnessed["post_valid"])
            self.assertIn("did not activate", harnessed["post_invalid_reason"])
            self.assertEqual(report["summary"]["harnessed"]["post_score"]["valid_count"], 0)
            self.assertEqual(report["summary"]["harnessed"]["comparison_valid"]["invalid"], 1)

    def test_harnessed_compare_can_require_recorded_mcp_tool_use(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template"
            template.mkdir()
            (template / "task.md").write_text("Optimize this.\n", encoding="utf-8")
            (template / "post_score.py").write_text(
                "import json\nprint(json.dumps({'best_cycles': 7}))\n",
                encoding="utf-8",
            )
            artifact_source = root / "artifact_source"
            mcp_dir = artifact_source / ".harnessgym" / "mcp" / "probe_server"
            mcp_dir.mkdir(parents=True)
            server = mcp_dir / "server.py"
            server.write_text(
                "\n".join(
                    [
                        "import json",
                        "import sys",
                        "",
                        "if '--self-test' in sys.argv:",
                        "    sys.exit(0)",
                        "",
                        "def read_msg():",
                        "    headers = {}",
                        "    while True:",
                        "        line = sys.stdin.buffer.readline()",
                        "        if line in (b'\\r\\n', b'\\n', b''):",
                        "            break",
                        "        key, _, value = line.decode('ascii').partition(':')",
                        "        headers[key.lower()] = value.strip()",
                        "    length = int(headers.get('content-length', '0'))",
                        "    return json.loads(sys.stdin.buffer.read(length).decode('utf-8'))",
                        "",
                        "def write_msg(payload):",
                        "    body = json.dumps(payload).encode('utf-8')",
                        "    sys.stdout.buffer.write(f'Content-Length: {len(body)}\\r\\n\\r\\n'.encode('ascii') + body)",
                        "    sys.stdout.buffer.flush()",
                        "",
                        "while True:",
                        "    msg = read_msg()",
                        "    if msg.get('method') == 'initialize':",
                        "        write_msg({'jsonrpc': '2.0', 'id': msg.get('id'), 'result': {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'probe', 'version': '1.0'}}})",
                        "    elif msg.get('method') == 'notifications/initialized':",
                        "        continue",
                        "    elif msg.get('method') == 'tools/list':",
                        "        write_msg({'jsonrpc': '2.0', 'id': msg.get('id'), 'result': {'tools': [{'name': 'probe_plan', 'description': 'Probe', 'inputSchema': {'type': 'object', 'properties': {}}}]}})",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (mcp_dir / "harnessgym-mcp.json").write_text(
                json.dumps(
                    {
                        "name": "probe-server",
                        "command": sys.executable,
                        "args": [".harnessgym/mcp/probe_server/server.py"],
                        "enabled_tools": ["probe_plan"],
                        "self_test": True,
                    }
                ),
                encoding="utf-8",
            )

            config = CompareConfig.from_values(
                workspace_template=template,
                output_dir=root / "compare",
                task="task.md",
                artifact_source=artifact_source,
                runner="fake",
                trials=1,
                post_command=f"{shlex.quote(sys.executable)} post_score.py",
                post_score_key="best_cycles",
                require_harness_tool_use=True,
            )
            report = run_compare(config)

            harnessed = next(trial for trial in report["trials"] if trial["group"] == "harnessed")
            self.assertEqual(harnessed["harness_validation"]["active_tool_count"], 1)
            self.assertEqual(harnessed["harness_validation"]["mcp_call_count"], 0)
            self.assertFalse(harnessed["comparison_valid"])
            self.assertIn("no MCP tool calls", harnessed["comparison_invalid_reason"])
            self.assertEqual(report["summary"]["harnessed"]["mcp_call_count"]["best"], 0.0)

    def test_failed_post_command_is_reported_as_invalid_worst_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template"
            template.mkdir()
            (template / "task.md").write_text("Optimize this.\n", encoding="utf-8")
            (template / "post_score.py").write_text(
                "import json, sys\nprint(json.dumps({'status': 'failed', 'best_cycles': 3}))\nsys.exit(1)\n",
                encoding="utf-8",
            )
            output_dir = root / "compare"

            config = CompareConfig.from_values(
                workspace_template=template,
                output_dir=output_dir,
                task="task.md",
                runner="fake",
                trials=1,
                post_command=f"{shlex.quote(sys.executable)} post_score.py",
                post_score_key="best_cycles",
            )
            report = run_compare(config)

            for trial in report["trials"]:
                self.assertFalse(trial["post_valid"])
                self.assertTrue(trial["post_treated_as_worst"])
                self.assertIsNone(trial["post_score"])
                self.assertIn("did not pass", trial["post_invalid_reason"])
            self.assertEqual(report["summary"]["plain"]["post_score"]["valid_count"], 0)
            self.assertEqual(report["summary"]["plain"]["post_score"]["invalid_count"], 1)
            self.assertTrue(report["summary"]["plain"]["post_score"]["invalid_treated_as_worst"])

    def test_failed_runner_attempt_invalidates_compare_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template"
            template.mkdir()
            (template / "task.md").write_text("Optimize this.\n", encoding="utf-8")
            (template / "post_score.py").write_text(
                "import json\nprint(json.dumps({'best_cycles': 7}))\n",
                encoding="utf-8",
            )

            config = CompareConfig.from_values(
                workspace_template=template,
                output_dir=root / "compare",
                task="task.md",
                runner="exec",
                codex_bin=str(root / "missing-codex"),
                trials=1,
                post_command=f"{shlex.quote(sys.executable)} post_score.py",
                post_score_key="best_cycles",
            )
            report = run_compare(config)

            for trial in report["trials"]:
                self.assertFalse(trial["comparison_valid"])
                self.assertFalse(trial["runner_validation"]["valid"])
                self.assertEqual(trial["runner_validation"]["failed_iterations"], [1])
                self.assertFalse(trial["post_valid"])
                self.assertIsNone(trial["post_score"])
                self.assertEqual(trial["iterations"][0]["attempt_return_code"], 127)
                self.assertIn("Codex executable not found", trial["iterations"][0]["attempt_message"])
                self.assertTrue(trial["iterations"][0]["attempt_transcript_path"].endswith("attempt.transcript.txt"))
                failed_attempt = trial["runner_validation"]["failed_attempts"][0]
                self.assertEqual(failed_attempt["iteration"], 1)
                self.assertEqual(failed_attempt["attempt_return_code"], 127)
                self.assertIn("Codex executable not found", failed_attempt["attempt_message"])
                self.assertTrue(failed_attempt["attempt_transcript_path"].endswith("attempt.transcript.txt"))
                self.assertIn("runner attempt failed", trial["comparison_invalid_reason"])
                self.assertIn("runner attempt failed", trial["post_invalid_reason"])
            self.assertEqual(report["summary"]["plain"]["post_score"]["valid_count"], 0)
            self.assertEqual(report["summary"]["plain"]["comparison_valid"]["invalid"], 1)
            self.assertIn(
                "runner attempt failed in iteration(s): 1",
                report["summary"]["plain"]["invalid_reasons"],
            )
