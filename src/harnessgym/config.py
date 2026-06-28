from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .timeouts import parse_timeout
from .task_state import TASK_STATE_CHOICES


RUNNER_CHOICES = ("exec", "claude", "tui-goal", "fake")
HARNESS_DEPTH_CHOICES = ("standard", "deep")


@dataclass(frozen=True)
class RunConfig:
    workspace: Path
    task_text: str
    task_path: Path | None
    iterations: int
    attempt_timeout_seconds: int
    build_timeout_seconds: int
    reflection_timeout_seconds: int
    attempt_timeout_schedule_seconds: tuple[int, ...] = ()
    runner: str = "exec"
    codex_bin: str = "codex"
    claude_bin: str = "claude"
    claude_model: str | None = None
    claude_permission_mode: str = "bypassPermissions"
    claude_max_budget_usd: float | None = None
    claude_extra_args: tuple[str, ...] = ()
    run_id: str | None = None
    build_after_solve: bool = False
    stop_score: float | None = None
    score_key: str = "score"
    higher_is_better: bool = False
    task_state: str = "continue"
    attempt_only: bool = False
    harness_depth: str = "deep"
    post_attempt_command: str | None = None
    post_attempt_score_key: str | None = None
    post_attempt_timeout_seconds: int = 120
    optimization_mode: bool = False
    restore_best: bool = True
    qualify_artifacts: bool = True
    artifact_repair_attempts: int = 1

    @classmethod
    def from_values(
        cls,
        *,
        workspace: str | Path,
        task: str | Path | None = None,
        task_text: str | None = None,
        iterations: int = 3,
        attempt_timeout: str | int | float = "45m",
        attempt_timeouts: str | None = None,
        build_timeout: str | int | float = "20m",
        reflection_timeout: str | int | float | None = None,
        runner: str = "exec",
        codex_bin: str = "codex",
        claude_bin: str = "claude",
        claude_model: str | None = None,
        claude_permission_mode: str = "bypassPermissions",
        claude_max_budget_usd: float | None = None,
        claude_extra_args: tuple[str, ...] | list[str] | None = None,
        run_id: str | None = None,
        build_after_solve: bool = False,
        stop_score: float | None = None,
        score_key: str = "score",
        higher_is_better: bool = False,
        task_state: str = "continue",
        attempt_only: bool = False,
        harness_depth: str = "deep",
        post_attempt_command: str | None = None,
        post_attempt_score_key: str | None = None,
        post_attempt_timeout: str | int | float = "2m",
        optimization_mode: bool = False,
        restore_best: bool = True,
        qualify_artifacts: bool = True,
        artifact_repair_attempts: int = 1,
    ) -> "RunConfig":
        if runner not in RUNNER_CHOICES:
            choices = ", ".join(RUNNER_CHOICES)
            raise ValueError(f"runner must be one of: {choices}")
        if task_state not in TASK_STATE_CHOICES:
            choices = ", ".join(TASK_STATE_CHOICES)
            raise ValueError(f"task_state must be one of: {choices}")
        if harness_depth not in HARNESS_DEPTH_CHOICES:
            choices = ", ".join(HARNESS_DEPTH_CHOICES)
            raise ValueError(f"harness_depth must be one of: {choices}")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if artifact_repair_attempts < 0:
            raise ValueError("artifact_repair_attempts cannot be negative")

        workspace_path = Path(workspace).resolve()
        task_path = None
        if task:
            raw_task_path = Path(task)
            if raw_task_path.is_absolute():
                task_path = raw_task_path.resolve()
            else:
                cwd_candidate = raw_task_path.resolve()
                workspace_candidate = (workspace_path / raw_task_path).resolve()
                task_path = cwd_candidate if cwd_candidate.exists() else workspace_candidate
        if task_text is None:
            if task_path is None:
                raise ValueError("provide either task text or a task file")
            task_text = task_path.read_text(encoding="utf-8")
        if not task_text.strip():
            raise ValueError("task text cannot be empty")

        build_seconds = parse_timeout(build_timeout)
        return cls(
            workspace=workspace_path,
            task_text=task_text,
            task_path=task_path,
            iterations=iterations,
            attempt_timeout_seconds=parse_timeout(attempt_timeout),
            build_timeout_seconds=build_seconds,
            reflection_timeout_seconds=parse_timeout(reflection_timeout)
            if reflection_timeout is not None
            else build_seconds,
            attempt_timeout_schedule_seconds=parse_timeout_list(attempt_timeouts),
            runner=runner,
            codex_bin=codex_bin,
            claude_bin=claude_bin,
            claude_model=claude_model,
            claude_permission_mode=claude_permission_mode,
            claude_max_budget_usd=claude_max_budget_usd,
            claude_extra_args=tuple(claude_extra_args or ()),
            run_id=run_id,
            build_after_solve=build_after_solve,
            stop_score=stop_score,
            score_key=score_key,
            higher_is_better=higher_is_better,
            task_state=task_state,
            attempt_only=attempt_only,
            harness_depth=harness_depth,
            post_attempt_command=post_attempt_command,
            post_attempt_score_key=post_attempt_score_key,
            post_attempt_timeout_seconds=parse_timeout(post_attempt_timeout),
            optimization_mode=optimization_mode,
            restore_best=restore_best,
            qualify_artifacts=qualify_artifacts,
            artifact_repair_attempts=artifact_repair_attempts,
        )

    def attempt_timeout_for(self, iteration: int) -> int:
        if not self.attempt_timeout_schedule_seconds:
            return self.attempt_timeout_seconds
        index = min(iteration - 1, len(self.attempt_timeout_schedule_seconds) - 1)
        return self.attempt_timeout_schedule_seconds[index]


def parse_timeout_list(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("attempt timeout schedule cannot be empty")
    return tuple(parse_timeout(piece) for piece in pieces)
