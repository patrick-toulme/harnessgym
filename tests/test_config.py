import tempfile
import unittest
from pathlib import Path

from harnessgym.cli import build_parser
from harnessgym.config import RunConfig


class ConfigTests(unittest.TestCase):
    def test_run_config_reads_task_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            task = tmp_path / "task.md"
            task.write_text("Fix the bug.\n", encoding="utf-8")

            config = RunConfig.from_values(
                workspace=tmp_path,
                task=task,
                iterations=2,
                attempt_timeout="2m",
                build_timeout="30s",
                runner="fake",
            )

            self.assertEqual(config.workspace, tmp_path.resolve())
            self.assertEqual(config.task_text, "Fix the bug.\n")
            self.assertEqual(config.task_path, task.resolve())
            self.assertEqual(config.iterations, 2)
            self.assertEqual(config.attempt_timeout_seconds, 120)
            self.assertEqual(config.build_timeout_seconds, 30)
            self.assertEqual(config.reflection_timeout_seconds, 30)
            self.assertEqual(config.runner, "fake")

    def test_attempt_timeout_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunConfig.from_values(
                workspace=Path(temp_dir),
                task_text="Task.",
                attempt_timeout="1m",
                attempt_timeouts="5s,2m",
                stop_score=0.5,
                score_key="best_ms",
                task_state="reset",
            )

            self.assertEqual(config.attempt_timeout_schedule_seconds, (5, 120))
            self.assertEqual(config.attempt_timeout_for(1), 5)
            self.assertEqual(config.attempt_timeout_for(2), 120)
            self.assertEqual(config.attempt_timeout_for(3), 120)
            self.assertEqual(config.stop_score, 0.5)
            self.assertEqual(config.score_key, "best_ms")
            self.assertEqual(config.task_state, "reset")
            self.assertEqual(config.harness_depth, "deep")
            self.assertTrue(config.restore_best)
            self.assertTrue(config.qualify_artifacts)
            self.assertEqual(config.artifact_repair_attempts, 1)
            self.assertEqual(config.claude_bin, "claude")
            self.assertEqual(config.claude_permission_mode, "bypassPermissions")

    def test_harness_depth_can_be_set_to_standard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunConfig.from_values(
                workspace=Path(temp_dir),
                task_text="Build tooling.",
                harness_depth="standard",
            )

            self.assertEqual(config.harness_depth, "standard")


    def test_cli_parser_accepts_inline_task(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--task-text",
                "Solve this.",
                "--workspace",
                ".",
                "--iterations",
                "3",
                "--attempt-timeout",
                "45m",
                "--build-timeout",
                "20m",
                "--runner",
                "claude",
                "--claude-bin",
                "claude-dev",
                "--claude-model",
                "sonnet",
                "--claude-permission-mode",
                "acceptEdits",
                "--claude-max-budget-usd",
                "2.5",
                "--claude-extra-arg=--verbose",
                "--post-attempt-command",
                "python3 verifier.py",
                "--post-attempt-score-key",
                "best_cycles",
                "--post-attempt-timeout",
                "3m",
                "--optimization-mode",
                "--no-restore-best",
                "--no-qualify-artifacts",
                "--artifact-repair-attempts",
                "2",
            ]
        )

        self.assertEqual(args.task_text, "Solve this.")
        self.assertEqual(args.iterations, 3)
        self.assertEqual(args.runner, "claude")
        self.assertEqual(args.claude_bin, "claude-dev")
        self.assertEqual(args.claude_model, "sonnet")
        self.assertEqual(args.claude_permission_mode, "acceptEdits")
        self.assertEqual(args.claude_max_budget_usd, 2.5)
        self.assertEqual(args.claude_extra_arg, ["--verbose"])
        self.assertEqual(args.harness_depth, "deep")
        self.assertEqual(args.post_attempt_command, "python3 verifier.py")
        self.assertEqual(args.post_attempt_score_key, "best_cycles")
        self.assertEqual(args.post_attempt_timeout, "3m")
        self.assertTrue(args.optimization_mode)
        self.assertFalse(args.restore_best)
        self.assertFalse(args.qualify_artifacts)
        self.assertEqual(args.artifact_repair_attempts, 2)

    def test_cli_parser_defaults_restore_best(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--task-text", "Solve this.", "--workspace", "."])

        self.assertTrue(args.restore_best)

    def test_compare_cli_parser_accepts_claude_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "compare",
                "--task-text",
                "Improve this.",
                "--workspace-template",
                "examples/tensor_layout_pipeline_task",
                "--artifact-source",
                "tmp/generated/.harnessgym",
                "--output-dir",
                "tmp/compare",
                "--runner",
                "claude",
                "--claude-bin",
                "claude-dev",
                "--claude-model",
                "sonnet",
                "--claude-permission-mode",
                "acceptEdits",
                "--claude-max-budget-usd",
                "3.25",
                "--claude-extra-arg=--verbose",
                "--post-command",
                "python3 benchmark.py --json --mode final",
                "--post-score-key",
                "best_cycles",
                "--no-require-active-harness",
                "--require-harness-tool-use",
            ]
        )

        self.assertEqual(args.task_text, "Improve this.")
        self.assertEqual(args.runner, "claude")
        self.assertEqual(args.claude_bin, "claude-dev")
        self.assertEqual(args.claude_model, "sonnet")
        self.assertEqual(args.claude_permission_mode, "acceptEdits")
        self.assertEqual(args.claude_max_budget_usd, 3.25)
        self.assertEqual(args.claude_extra_arg, ["--verbose"])
        self.assertEqual(args.post_command, "python3 benchmark.py --json --mode final")
        self.assertEqual(args.post_score_key, "best_cycles")
        self.assertFalse(args.require_active_harness)
        self.assertTrue(args.require_harness_tool_use)

    def test_relative_task_can_resolve_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            task = workspace / "task.md"
            task.write_text("Workspace task.\n", encoding="utf-8")

            config = RunConfig.from_values(workspace=workspace, task="task.md")

            self.assertEqual(config.task_path, task.resolve())
            self.assertEqual(config.task_text, "Workspace task.\n")
