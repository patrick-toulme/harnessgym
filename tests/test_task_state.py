import json
import tempfile
import unittest
from pathlib import Path

from harnessgym.config import RunConfig
from harnessgym.models import IterationContext, RunnerResult
from harnessgym.orchestrator import Orchestrator
from harnessgym.runners.base import Runner


class MutatingRunner(Runner):
    def __init__(self) -> None:
        self.second_prompt = ""

    def start_attempt(self, config: RunConfig, context: IterationContext, prompt: str) -> RunnerResult:
        marker = context.workspace / "work.txt"
        marker.write_text(f"iteration-{context.iteration}\n", encoding="utf-8")
        if context.iteration == 1:
            tool = context.workspace / ".harnessgym" / "tools" / "marker.txt"
            tool.parent.mkdir(parents=True, exist_ok=True)
            tool.write_text("tool from iteration 1\n", encoding="utf-8")
        else:
            self.second_prompt = prompt
        context.result_path.write_text(
            json.dumps(
                {
                    "status": "blocked" if context.iteration == 1 else "solved",
                    "verified": context.iteration == 2,
                    "metrics": {"score": 1.0},
                }
            ),
            encoding="utf-8",
        )
        return self._result(context, "attempt")

    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        return self._result(context, "reflection", session_id=session_id)

    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        return self._result(context, "build", session_id=session_id)

    def _result(
        self,
        context: IterationContext,
        phase: str,
        session_id: str | None = None,
    ) -> RunnerResult:
        path = context.iteration_dir / f"{phase}.txt"
        path.write_text(phase, encoding="utf-8")
        return RunnerResult(
            phase=phase,
            status="completed",
            session_id=session_id or f"session-{context.iteration}",
            stdout_path=str(path),
            stderr_path=str(path),
            transcript_path=str(path),
            prompt_path=str(path),
        )


class TaskStateTests(unittest.TestCase):
    def test_reset_restores_task_files_but_keeps_harness_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "task.md").write_text("Task.\n", encoding="utf-8")
            (workspace / "work.txt").write_text("initial\n", encoding="utf-8")
            runner = MutatingRunner()
            config = RunConfig.from_values(
                workspace=workspace,
                task="task.md",
                iterations=2,
                runner="fake",
                task_state="reset",
            )

            results = Orchestrator(runner=runner).run(config)

            self.assertEqual(len(results), 2)
            self.assertEqual((workspace / "work.txt").read_text(encoding="utf-8"), "iteration-2\n")
            self.assertIn(".harnessgym/tools/marker.txt", runner.second_prompt)
            snapshot = workspace / ".harnessgym" / "task_state" / "initial" / "work.txt"
            self.assertEqual(snapshot.read_text(encoding="utf-8"), "initial\n")

    def test_continue_keeps_previous_task_file_edits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "task.md").write_text("Task.\n", encoding="utf-8")
            (workspace / "work.txt").write_text("initial\n", encoding="utf-8")
            runner = MutatingRunner()
            config = RunConfig.from_values(
                workspace=workspace,
                task="task.md",
                iterations=2,
                runner="fake",
                task_state="continue",
            )

            Orchestrator(runner=runner).run(config)

            self.assertEqual((workspace / "work.txt").read_text(encoding="utf-8"), "iteration-2\n")
            self.assertFalse((workspace / ".harnessgym" / "task_state" / "initial").exists())
