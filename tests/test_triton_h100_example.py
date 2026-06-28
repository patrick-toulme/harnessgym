import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harnessgym.activation import activate_generated_harness
from harnessgym.compare import copy_harness_artifacts, copy_workspace_template
from harnessgym.registry import load_registry


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "triton_rmsnorm_h100_task"
HARNESS_ARTIFACTS = ROOT / "examples" / "triton_rmsnorm_h100_harness_artifacts" / ".harnessgym"


class TritonH100ExampleTests(unittest.TestCase):
    def test_task_template_contains_remote_h100_entrypoints(self) -> None:
        task = (EXAMPLE / "task.md").read_text(encoding="utf-8")
        config = json.loads((EXAMPLE / "kernel_config.json").read_text(encoding="utf-8"))

        self.assertIn("remote_h100.py", task)
        self.assertIn("python3 verifier.py --json --mode final", task)
        self.assertEqual(config["num_warps"], 1)
        self.assertEqual(config["num_stages"], 4)

    def test_remote_wrapper_is_syntax_valid_without_gpu_dependencies(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(EXAMPLE / "remote_h100.py")],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_committed_h100_harness_artifact_bundle_activates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            copy_workspace_template(EXAMPLE, workspace)
            copied = copy_harness_artifacts(HARNESS_ARTIFACTS, workspace)
            registry = load_registry(workspace)

            activation = activate_generated_harness(workspace, registry)

            self.assertIn(".harnessgym/mcp/h100_triton_rmsnorm/harnessgym-mcp.json", copied)
            self.assertEqual(activation["quality_gate"]["status"], "passed")
            self.assertEqual(activation["quality_gate"]["active_mcp_count"], 1)
            self.assertEqual(activation["quality_gate"]["active_tool_count"], 17)
            self.assertTrue((workspace / ".agents/skills/h100-triton-rmsnorm/SKILL.md").exists())

    def test_committed_h100_mcp_tests_pass_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            copy_workspace_template(EXAMPLE, workspace)
            copy_harness_artifacts(HARNESS_ARTIFACTS, workspace)

            result = subprocess.run(
                [sys.executable, ".harnessgym/tests/test_h100_triton_mcp.py"],
                cwd=workspace,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
