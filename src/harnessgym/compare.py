from __future__ import annotations

import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ARTIFACT_DIRS, EXCLUDED_TEMPLATE_NAMES, EXCLUDED_TEMPLATE_SUFFIXES, read_json, write_json
from .config import RunConfig
from .models import utc_now
from .mcp_telemetry import summarize_mcp_call_events
from .orchestrator import Orchestrator, extract_score
from .postprocess import post_command_passed, run_post_command
from .timeouts import parse_timeout


@dataclass(frozen=True)
class CompareConfig:
    workspace_template: Path
    output_dir: Path
    task: str | Path | None = None
    task_text: str | None = None
    artifact_source: Path | None = None
    trials: int = 1
    iterations: int = 1
    attempt_timeout: str = "5m"
    attempt_timeouts: str | None = None
    build_timeout: str = "1s"
    reflection_timeout: str | None = "1s"
    runner: str = "exec"
    codex_bin: str = "codex"
    claude_bin: str = "claude"
    claude_model: str | None = None
    claude_permission_mode: str = "bypassPermissions"
    claude_max_budget_usd: float | None = None
    claude_extra_args: tuple[str, ...] = ()
    stop_score: float | None = None
    score_key: str = "score"
    higher_is_better: bool = False
    task_state: str = "reset"
    build_after_solve: bool = False
    post_command: str | None = None
    post_score_key: str = "score"
    post_timeout: str = "2m"
    require_active_harness: bool = True
    require_harness_tool_use: bool = False
    overwrite: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        workspace_template: str | Path,
        output_dir: str | Path,
        task: str | Path | None = None,
        task_text: str | None = None,
        artifact_source: str | Path | None = None,
        trials: int = 1,
        iterations: int = 1,
        attempt_timeout: str = "5m",
        attempt_timeouts: str | None = None,
        build_timeout: str = "1s",
        reflection_timeout: str | None = "1s",
        runner: str = "exec",
        codex_bin: str = "codex",
        claude_bin: str = "claude",
        claude_model: str | None = None,
        claude_permission_mode: str = "bypassPermissions",
        claude_max_budget_usd: float | None = None,
        claude_extra_args: tuple[str, ...] | list[str] | None = None,
        stop_score: float | None = None,
        score_key: str = "score",
        higher_is_better: bool = False,
        task_state: str = "reset",
        build_after_solve: bool = False,
        post_command: str | None = None,
        post_score_key: str = "score",
        post_timeout: str = "2m",
        require_active_harness: bool = True,
        require_harness_tool_use: bool = False,
        overwrite: bool = False,
    ) -> "CompareConfig":
        if trials <= 0:
            raise ValueError("trials must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if task is None and task_text is None:
            raise ValueError("provide either task text or a task file")
        template_path = Path(workspace_template).expanduser().resolve()
        if not template_path.is_dir():
            raise ValueError(f"workspace template does not exist: {template_path}")
        source_path = Path(artifact_source).expanduser().resolve() if artifact_source else None
        if source_path is not None and not source_path.exists():
            raise ValueError(f"artifact source does not exist: {source_path}")
        return cls(
            workspace_template=template_path,
            output_dir=Path(output_dir).expanduser().resolve(),
            task=task,
            task_text=task_text,
            artifact_source=source_path,
            trials=trials,
            iterations=iterations,
            attempt_timeout=attempt_timeout,
            attempt_timeouts=attempt_timeouts,
            build_timeout=build_timeout,
            reflection_timeout=reflection_timeout,
            runner=runner,
            codex_bin=codex_bin,
            claude_bin=claude_bin,
            claude_model=claude_model,
            claude_permission_mode=claude_permission_mode,
            claude_max_budget_usd=claude_max_budget_usd,
            claude_extra_args=tuple(claude_extra_args or ()),
            stop_score=stop_score,
            score_key=score_key,
            higher_is_better=higher_is_better,
            task_state=task_state,
            build_after_solve=build_after_solve,
            post_command=post_command,
            post_score_key=post_score_key,
            post_timeout=post_timeout,
            require_active_harness=require_active_harness,
            require_harness_tool_use=require_harness_tool_use,
            overwrite=overwrite,
        )


def run_compare(config: CompareConfig) -> dict[str, Any]:
    """Run plain-vs-harnessed replay trials and write a machine-readable report."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    for group in ("plain", "harnessed"):
        for trial_index in range(1, config.trials + 1):
            trial = _run_trial(config, group, trial_index)
            trials.append(trial)

    report = {
        "created_at": utc_now(),
        "workspace_template": str(config.workspace_template),
        "artifact_source": str(config.artifact_source) if config.artifact_source else None,
        "trials": trials,
        "summary": _summarize_trials(trials, config.higher_is_better),
    }
    write_json(config.output_dir / "compare_report.json", report)
    return report


def copy_workspace_template(source: Path, destination: Path) -> list[str]:
    if destination.exists():
        raise FileExistsError(f"workspace destination already exists: {destination}")
    shutil.copytree(source, destination, ignore=_template_ignore)
    copied = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            copied.append(path.relative_to(destination).as_posix())
    return copied


def copy_harness_artifacts(artifact_source: Path, workspace: Path) -> list[str]:
    source_harness = _resolve_harness_dir(artifact_source)
    destination_harness = workspace / ".harnessgym"
    destination_harness.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for dirname in ARTIFACT_DIRS.values():
        source = source_harness / dirname
        if not source.exists():
            continue
        destination = destination_harness / dirname
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, ignore=_artifact_ignore)
        copied.extend(
            path.relative_to(workspace).as_posix()
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        )

    registry = source_harness / "registry.json"
    if registry.exists():
        destination = destination_harness / "registry.json"
        shutil.copy2(registry, destination)
        copied.append(destination.relative_to(workspace).as_posix())

    return sorted(copied)


def _run_trial(config: CompareConfig, group: str, trial_index: int) -> dict[str, Any]:
    trial_root = config.output_dir / group / f"trial_{trial_index}"
    if trial_root.exists():
        if not config.overwrite:
            raise FileExistsError(f"trial directory already exists: {trial_root}")
        shutil.rmtree(trial_root)
    workspace = trial_root / "workspace"
    trial_root.mkdir(parents=True, exist_ok=True)
    copy_workspace_template(config.workspace_template, workspace)

    copied_artifacts: list[str] = []
    if group == "harnessed" and config.artifact_source is not None:
        copied_artifacts = copy_harness_artifacts(config.artifact_source, workspace)

    task = _task_for_workspace(config.task, config.workspace_template, workspace)
    run_id = f"compare-{group}-{trial_index}"
    run_config = RunConfig.from_values(
        workspace=workspace,
        task=task,
        task_text=config.task_text,
        iterations=config.iterations,
        attempt_timeout=config.attempt_timeout,
        attempt_timeouts=config.attempt_timeouts,
        build_timeout=config.build_timeout,
        reflection_timeout=config.reflection_timeout,
        runner=config.runner,
        codex_bin=config.codex_bin,
        claude_bin=config.claude_bin,
        claude_model=config.claude_model,
        claude_permission_mode=config.claude_permission_mode,
        claude_max_budget_usd=config.claude_max_budget_usd,
        claude_extra_args=config.claude_extra_args,
        run_id=run_id,
        build_after_solve=config.build_after_solve,
        stop_score=config.stop_score,
        score_key=config.score_key,
        higher_is_better=config.higher_is_better,
        task_state=config.task_state,
        attempt_only=True,
        qualify_artifacts=False,
    )

    orchestrator = Orchestrator()
    error = ""
    started = time.monotonic()
    try:
        results = orchestrator.run(run_config)
    except Exception as exc:
        results = []
        error = str(exc)
    duration = time.monotonic() - started

    final = results[-1] if results else None
    result_path = Path(final.result_path) if final else None
    result_data = read_json(result_path) if result_path else {}
    iterations = _iteration_rows(results, config.score_key)
    mcp_telemetry = summarize_mcp_call_events(workspace)
    cumulative_attempt_duration = sum(
        row["attempt_duration_seconds"] for row in iterations if row["attempt_duration_seconds"] is not None
    )
    post_result = (
        run_post_command(
            command=config.post_command,
            cwd=workspace,
            log_dir=trial_root,
            timeout_seconds=parse_timeout(config.post_timeout),
            score_key=config.post_score_key,
        )
        if config.post_command
        else None
    )
    objective_post_valid = _post_result_valid(post_result)
    harness_validation = _harness_validation(group, copied_artifacts, iterations, mcp_telemetry, config)
    runner_validation = _runner_validation(error, iterations)
    comparison_valid = harness_validation["valid"] and runner_validation["valid"]
    post_valid = bool(objective_post_valid and comparison_valid)
    post_score = post_result.get("score") if post_result and post_valid else None
    comparison_invalid_reason = "" if comparison_valid else _comparison_invalid_reason(harness_validation, runner_validation)
    post_invalid_reason = ""
    if post_result and not post_valid:
        if not comparison_valid:
            post_invalid_reason = comparison_invalid_reason
        else:
            post_invalid_reason = _post_invalid_reason(post_result)
    trial = {
        "group": group,
        "trial": trial_index,
        "workspace": str(workspace),
        "run_id": run_id,
        "run_dir": str(orchestrator.run_dir) if orchestrator.run_dir else None,
        "result_path": str(result_path) if result_path else None,
        "status": final.status if final else "failed",
        "solved": bool(final and final.solved),
        "error": error,
        "duration_seconds": duration,
        "attempt_duration_seconds": final.attempt.duration_seconds if final and final.attempt else None,
        "cumulative_attempt_duration_seconds": cumulative_attempt_duration,
        "iterations_completed": len(iterations),
        "iterations": iterations,
        "score": extract_score(result_data, config.score_key),
        "copied_artifacts": copied_artifacts,
        "mcp_telemetry": mcp_telemetry,
        "mcp_call_count": mcp_telemetry["count"],
        "mcp_called_tools": mcp_telemetry["called_tools"],
        "runner_validation": runner_validation,
        "harness_validation": harness_validation,
        "comparison_valid": comparison_valid,
        "comparison_invalid_reason": comparison_invalid_reason,
        "post_result": post_result,
        "objective_post_valid": objective_post_valid if post_result else None,
        "post_valid": post_valid if post_result else None,
        "post_invalid_reason": post_invalid_reason,
        "post_score": post_score,
        "post_treated_as_worst": bool(post_result and not post_valid),
    }
    write_json(trial_root / "trial.json", trial)
    return trial


def _iteration_rows(results: list[Any], score_key: str) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        attempt = result.attempt
        result_data = read_json(Path(result.result_path))
        harness_usage = result_data.get("harness_usage") if isinstance(result_data, dict) else {}
        if not isinstance(harness_usage, dict):
            harness_usage = {}
        rows.append(
            {
                "iteration": result.iteration,
                "status": result.status,
                "solved": result.solved,
                "result_path": result.result_path,
                "attempt_status": attempt.status if attempt else None,
                "attempt_return_code": attempt.return_code if attempt else None,
                "attempt_timed_out": attempt.timed_out if attempt else None,
                "attempt_duration_seconds": attempt.duration_seconds if attempt else None,
                "attempt_message": attempt.message if attempt else "",
                "attempt_transcript_path": attempt.transcript_path if attempt else None,
                "score": extract_score(result_data, score_key),
                "active_mcp_count": harness_usage.get("active_mcp_count"),
                "active_tool_count": harness_usage.get("active_tool_count"),
                "used_active_tools": harness_usage.get("used_active_tools"),
                "mcp_call_count": harness_usage.get("mcp_call_count"),
                "mcp_called_tools": harness_usage.get("mcp_called_tools"),
            }
        )
    return rows


def _template_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(directory) / name
        if name in EXCLUDED_TEMPLATE_NAMES:
            ignored.add(name)
        elif path.is_file() and (path.suffix in EXCLUDED_TEMPLATE_SUFFIXES or name.startswith("flash_bench")):
            ignored.add(name)
    return ignored


def _artifact_ignore(directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__" or (Path(directory) / name).suffix in {".pyc", ".pyo"}
    }


def _resolve_harness_dir(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.name == ".harnessgym":
        return candidate
    nested = candidate / ".harnessgym"
    if nested.is_dir():
        return nested
    raise ValueError(f"artifact source is not a .harnessgym directory or workspace: {path}")


def _task_for_workspace(task: str | Path | None, template: Path, workspace: Path) -> Path | None:
    if task is None:
        return None
    raw = Path(task)
    if raw.is_absolute():
        return _map_task_path(raw.resolve(), template, workspace)
    cwd_candidate = raw.resolve()
    if cwd_candidate.exists():
        return _map_task_path(cwd_candidate, template, workspace)
    return workspace / raw


def _map_task_path(path: Path, template: Path, workspace: Path) -> Path:
    try:
        relative = path.relative_to(template.resolve())
    except ValueError:
        return path
    return workspace / relative


def _summarize_trials(trials: list[dict[str, Any]], higher_is_better: bool) -> dict[str, Any]:
    return {
        group: {
            "count": len(group_trials),
            "attempt_duration_seconds": _number_summary(
                [trial.get("attempt_duration_seconds") for trial in group_trials],
                higher_is_better=False,
            ),
            "cumulative_attempt_duration_seconds": _number_summary(
                [trial.get("cumulative_attempt_duration_seconds") for trial in group_trials],
                higher_is_better=False,
            ),
            "iterations_completed": _number_summary(
                [trial.get("iterations_completed") for trial in group_trials],
                higher_is_better=True,
            ),
            "score": _number_summary([trial.get("score") for trial in group_trials], higher_is_better),
            "post_score": _post_score_summary(
                group_trials,
                higher_is_better,
            ),
            "post_status": _status_counts(
                [trial.get("post_result", {}).get("status") for trial in group_trials]
            ),
            "comparison_valid": _status_counts(
                ["valid" if trial.get("comparison_valid") else "invalid" for trial in group_trials]
            ),
            "mcp_call_count": _number_summary(
                [trial.get("mcp_call_count") for trial in group_trials],
                higher_is_better=True,
            ),
            "mcp_called_tools": sorted(
                {
                    str(tool)
                    for trial in group_trials
                    for tool in (trial.get("mcp_called_tools") or [])
                }
            ),
            "invalid_reasons": _status_counts(
                [
                    trial.get("comparison_invalid_reason")
                    for trial in group_trials
                    if not trial.get("comparison_valid")
                ]
            ),
        }
        for group in ("plain", "harnessed")
        for group_trials in [[trial for trial in trials if trial.get("group") == group]]
    }


def _number_summary(values: list[Any], higher_is_better: bool) -> dict[str, Any]:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return {"count": 0, "min": None, "median": None, "max": None, "best": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "median": statistics.median(numbers),
        "max": max(numbers),
        "best": max(numbers) if higher_is_better else min(numbers),
    }


def _post_score_summary(group_trials: list[dict[str, Any]], higher_is_better: bool) -> dict[str, Any]:
    post_trials = [trial for trial in group_trials if trial.get("post_result") is not None]
    valid_scores = [
        trial.get("post_score")
        for trial in post_trials
        if trial.get("post_valid") and isinstance(trial.get("post_score"), (int, float))
    ]
    summary = _number_summary(valid_scores, higher_is_better)
    invalid_count = sum(1 for trial in post_trials if trial.get("post_valid") is False)
    not_run_count = len(group_trials) - len(post_trials)
    summary.update(
        {
            "valid_count": len(valid_scores),
            "invalid_count": invalid_count,
            "not_run_count": not_run_count,
            "invalid_treated_as_worst": invalid_count > 0,
        }
    )
    return summary


def _status_counts(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "not_run")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _post_result_valid(post_result: dict[str, Any] | None) -> bool:
    if not post_result:
        return False
    return post_command_passed(post_result) and isinstance(post_result.get("score"), (int, float))


def _post_invalid_reason(post_result: dict[str, Any]) -> str:
    if not post_command_passed(post_result):
        return f"post command did not pass: status={post_result.get('status')} return_code={post_result.get('return_code')}"
    if not isinstance(post_result.get("score"), (int, float)):
        return f"post command did not emit numeric score for key {post_result.get('score_key')}"
    return ""


def _runner_validation(error: str, iterations: list[dict[str, Any]]) -> dict[str, Any]:
    validation: dict[str, Any] = {
        "valid": True,
        "reason": "",
        "failed_iterations": [],
        "failed_attempts": [],
    }
    if error:
        validation["valid"] = False
        validation["reason"] = f"runner trial raised exception: {error}"
        return validation
    if not iterations:
        validation["valid"] = False
        validation["reason"] = "runner trial invalid: no iterations completed"
        return validation
    failed_rows = [
        row
        for row in iterations
        if row.get("attempt_status") == "failed" or row.get("status") == "failed"
    ]
    failed_iterations = [int(row["iteration"]) for row in failed_rows]
    if failed_iterations:
        validation["valid"] = False
        validation["failed_iterations"] = failed_iterations
        validation["failed_attempts"] = [
            {
                "iteration": int(row["iteration"]),
                "status": row.get("status"),
                "attempt_status": row.get("attempt_status"),
                "attempt_return_code": row.get("attempt_return_code"),
                "attempt_timed_out": row.get("attempt_timed_out"),
                "attempt_message": row.get("attempt_message") or "",
                "attempt_transcript_path": row.get("attempt_transcript_path"),
            }
            for row in failed_rows
        ]
        validation["reason"] = "runner attempt failed in iteration(s): " + ", ".join(
            str(iteration) for iteration in failed_iterations
        )
    return validation


def _comparison_invalid_reason(
    harness_validation: dict[str, Any],
    runner_validation: dict[str, Any],
) -> str:
    reasons = [
        str(validation.get("reason"))
        for validation in (runner_validation, harness_validation)
        if validation.get("reason")
    ]
    return "; ".join(reasons)


def _harness_validation(
    group: str,
    copied_artifacts: list[str],
    iterations: list[dict[str, Any]],
    mcp_telemetry: dict[str, Any],
    config: CompareConfig,
) -> dict[str, Any]:
    if group != "harnessed":
        return {"required": False, "valid": True, "reason": ""}
    active_required = bool(config.require_active_harness and copied_artifacts)
    tool_use_required = bool(config.require_harness_tool_use and copied_artifacts)
    best_active_tool_count = max(
        [int(row.get("active_tool_count") or 0) for row in iterations] or [0]
    )
    best_active_mcp_count = max(
        [int(row.get("active_mcp_count") or 0) for row in iterations] or [0]
    )
    validation = {
        "required": active_required or tool_use_required,
        "active_harness_required": active_required,
        "tool_use_required": tool_use_required,
        "valid": True,
        "reason": "",
        "active_tool_count": best_active_tool_count,
        "active_mcp_count": best_active_mcp_count,
        "mcp_call_count": int(mcp_telemetry.get("count") or 0),
        "mcp_successful_call_count": int(mcp_telemetry.get("successful_count") or 0),
        "mcp_called_tools": list(mcp_telemetry.get("called_tools") or []),
        "mcp_call_status_counts": dict(mcp_telemetry.get("status_counts") or {}),
    }
    if active_required and best_active_tool_count <= 0:
        validation["valid"] = False
        validation["reason"] = (
            "harnessed trial invalid: copied artifacts did not activate any generated MCP tools"
        )
    elif tool_use_required and int(mcp_telemetry.get("count") or 0) <= 0:
        validation["valid"] = False
        validation["reason"] = (
            "harnessed trial invalid: generated MCP tools activated but no MCP tool calls were recorded"
        )
    return validation
