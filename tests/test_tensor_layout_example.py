import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harnessgym.activation import activate_generated_harness
from harnessgym.compare import copy_harness_artifacts, copy_workspace_template
from harnessgym.registry import load_registry
from harnessgym.runners.claude_runner import ClaudeRunner


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tensor_layout_pipeline_task"
HARNESS_ARTIFACTS = ROOT / "examples" / "tensor_layout_harness_artifacts" / ".harnessgym"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("tensor_layout_benchmark", EXAMPLE / "benchmark.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class TensorLayoutExampleTests(unittest.TestCase):
    def test_starting_plan_passes_dev_and_final_with_headroom(self) -> None:
        benchmark = load_benchmark_module()
        plan = benchmark.load_plan(EXAMPLE / "kernel_plan.json")

        dev = benchmark.evaluate_plan(plan, "dev")
        final = benchmark.evaluate_plan(plan, "final")

        self.assertEqual(dev["status"], "passed")
        self.assertEqual(final["status"], "passed")
        self.assertGreater(final["best_cycles"], dev["best_cycles"])
        self.assertGreater(final["best_cycles"], 10_000_000)

    def test_trace_contains_layout_dma_and_pressure_signals(self) -> None:
        benchmark = load_benchmark_module()
        plan = benchmark.load_plan(EXAMPLE / "kernel_plan.json")
        trace_case = benchmark.case_breakdown(plan, benchmark.CASES["dev"][0])

        self.assertIn("components", trace_case)
        self.assertIn("bank_conflicts", trace_case)
        self.assertIn("dma_penalties", trace_case)
        self.assertIn("register_pressure", trace_case)
        self.assertIn("descriptor_count", trace_case)

    def test_verifier_cli_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "verifier.py"],
            cwd=EXAMPLE,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"status": "passed"', result.stdout)

    def test_trace_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.json"
            result = subprocess.run(
                [sys.executable, "benchmark.py", "--json", "--mode", "dev", "--trace", str(trace_path)],
                cwd=EXAMPLE,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(trace_path.exists())
            self.assertIn('"valid_values"', trace_path.read_text(encoding="utf-8"))

    def test_committed_harness_artifact_bundle_activates_for_claude(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            copy_workspace_template(EXAMPLE, workspace)
            copied = copy_harness_artifacts(HARNESS_ARTIFACTS, workspace)
            registry = load_registry(workspace)

            activation = activate_generated_harness(workspace, registry)
            command = ClaudeRunner(claude_bin="claude").build_command("attempt", workspace=workspace)

            self.assertIn(".harnessgym/mcp/tensor-plan-server/tensor_plan_server.py", copied)
            self.assertEqual(activation["quality_gate"]["status"], "passed")
            self.assertEqual(activation["quality_gate"]["active_mcp_count"], 1)
            self.assertEqual(activation["quality_gate"]["active_tool_count"], 15)
            self.assertIn("--mcp-config", command)
            self.assertIn("--allowedTools=mcp__tensor-plan-server", command)
            config_path = workspace / ".harnessgym" / "claude_mcp_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            server = config["mcpServers"]["tensor-plan-server"]
            self.assertEqual(server["command"], sys.executable)
            self.assertIn("claude_mcp_bridge.py", server["args"][0])
            self.assertEqual(server["env"]["HARNESSGYM_WORKSPACE"], str(workspace))

    def test_claude_compare_helper_uses_committed_artifact_bundle(self) -> None:
        helper = ROOT / "examples" / "tensor_layout_harness_artifacts" / "run_claude_compare.sh"
        text = helper.read_text(encoding="utf-8")

        self.assertTrue(helper.exists())
        self.assertIn('"$CLAUDE_BIN" -p --output-format json "Reply OK."', text)
        self.assertIn("Claude Code preflight failed; not running compare.", text)
        self.assertIn("api_error_status", text)
        self.assertIn("WAIT_FOR_CLAUDE", text)
        self.assertIn("PREFLIGHT_MAX_WAIT_SECONDS", text)
        self.assertIn("check_claude_compare_report.py", text)
        self.assertIn("--artifact-source examples/tensor_layout_harness_artifacts/.harnessgym", text)
        self.assertIn("--workspace-template examples/tensor_layout_pipeline_task", text)
        self.assertIn("--runner claude", text)

    def test_claude_compare_report_checker_accepts_valid_report(self) -> None:
        checker = ROOT / "examples" / "tensor_layout_harness_artifacts" / "check_claude_compare_report.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "compare_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "trials": [
                            {
                                "group": "plain",
                                "trial": 1,
                                "comparison_valid": True,
                                "post_result": {"status": "passed"},
                                "post_valid": True,
                                "post_score": 123.0,
                            },
                            {
                                "group": "harnessed",
                                "trial": 1,
                                "comparison_valid": True,
                                "post_result": {"status": "passed"},
                                "post_valid": True,
                                "post_score": 99.0,
                                "copied_artifacts": [".harnessgym/mcp/tensor-plan-server/harnessgym-mcp.json"],
                                "harness_validation": {
                                    "active_mcp_count": 1,
                                    "active_tool_count": 15,
                                },
                                "iterations": [
                                    {
                                        "active_mcp_count": 1,
                                        "active_tool_count": 15,
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(checker),
                    str(report_path),
                    "--min-active-mcp",
                    "1",
                    "--min-active-tools",
                    "15",
                    "--require-harness-win",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["harnessed_best_active_tool_count"], 15)

    def test_claude_compare_report_checker_rejects_missing_harness_tools(self) -> None:
        checker = ROOT / "examples" / "tensor_layout_harness_artifacts" / "check_claude_compare_report.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "compare_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "trials": [
                            {
                                "group": "plain",
                                "trial": 1,
                                "comparison_valid": True,
                                "post_result": {"status": "passed"},
                                "post_valid": True,
                                "post_score": 123.0,
                            },
                            {
                                "group": "harnessed",
                                "trial": 1,
                                "comparison_valid": True,
                                "post_result": {"status": "passed"},
                                "post_valid": True,
                                "post_score": 99.0,
                                "copied_artifacts": [],
                                "harness_validation": {
                                    "active_mcp_count": 0,
                                    "active_tool_count": 0,
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(checker), str(report_path), "--min-active-tools", "15"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "failed")
            self.assertTrue(any("activated 0 tools" in error for error in payload["errors"]))


if __name__ == "__main__":
    unittest.main()
