from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harnessgym.config import RunConfig
from harnessgym.orchestrator import Orchestrator
from harnessgym.registry import load_registry


def copy_demo_workspace(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "examples" / "numerical_debug_task"
    workspace = tmp_path / "numerical_debug_task"
    shutil.copytree(source, workspace)
    return workspace


class FakeE2ETests(unittest.TestCase):
    def test_fake_runner_end_to_end_two_iterations_and_artifact_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = copy_demo_workspace(Path(temp_dir))
            config = RunConfig.from_values(
                workspace=workspace,
                task=workspace / "task.md",
                iterations=2,
                attempt_timeout="5s",
                build_timeout="5s",
                runner="fake",
                run_id="fake-e2e",
            )

            results = Orchestrator().run(config)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].status, "tooling_built")
            self.assertEqual(results[1].status, "solved")
            self.assertTrue(results[1].solved)

            registry = load_registry(workspace)
            artifact = registry.get_artifact("tool:.harnessgym/tools/harnessgym_fake_probe.py")
            self.assertIsNotNone(artifact)
            assert artifact is not None
            self.assertTrue((workspace / artifact.path).exists())

            iteration1 = workspace / ".harnessgym" / "runs" / "fake-e2e" / "iterations" / "1"
            iteration2 = workspace / ".harnessgym" / "runs" / "fake-e2e" / "iterations" / "2"
            self.assertTrue((iteration1 / "result.json").exists())
            self.assertTrue((iteration1 / "attempt.transcript.txt").exists())
            self.assertTrue((iteration1 / "build.stdout.txt").exists())
            self.assertTrue((iteration2 / "result.json").exists())

            attempt2_prompt = (iteration2 / "attempt.prompt.txt").read_text(encoding="utf-8")
            self.assertIn(".harnessgym/tools/harnessgym_fake_probe.py", attempt2_prompt)
            self.assertIn("Fake debugging probe", attempt2_prompt)

            verifier = subprocess.run(
                [sys.executable, "verifier.py"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(verifier.returncode, 0, verifier.stdout + verifier.stderr)
