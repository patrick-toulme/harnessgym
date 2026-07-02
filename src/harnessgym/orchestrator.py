from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .artifacts import (
    create_initial_result,
    ensure_harness_dirs,
    iteration_dir,
    make_run_id,
    read_json,
    run_dir,
    sync_registry_from_files,
    update_result,
    write_json,
)
from .activation import activate_generated_harness
from .checkpoints import BestCheckpointManager
from .config import RunConfig
from .models import IterationContext, IterationResult, RunnerResult
from .mcp_telemetry import summarize_mcp_call_events
from .postprocess import (
    extract_score_from_payload,
    post_command_passed,
    run_post_command,
)
from .prompts import (
    extract_selected_improvement,
    render_artifact_context,
    render_attempt_prompt,
    render_build_prompt,
    render_reflection_prompt,
    render_repair_prompt,
)
from .qualification import capture_clean_workspace_template, qualify_generated_harness
from .registry import (
    load_registry,
    mark_artifacts_qualified,
    quarantine_artifacts,
    save_registry,
)
from .runners import ClaudeRunner, ExecRunner, FakeRunner, Runner, TuiGoalRunner
from .task_state import TaskStateManager


def make_runner(config: RunConfig) -> Runner:
    if config.runner == "exec":
        return ExecRunner(codex_bin=config.codex_bin)
    if config.runner == "claude":
        return ClaudeRunner(
            claude_bin=config.claude_bin,
            model=config.claude_model,
            permission_mode=config.claude_permission_mode,
            max_budget_usd=config.claude_max_budget_usd,
            extra_args=config.claude_extra_args,
        )
    if config.runner == "tui-goal":
        return TuiGoalRunner(codex_bin=config.codex_bin)
    if config.runner == "fake":
        return FakeRunner()
    raise ValueError(f"unsupported runner: {config.runner}")


class Orchestrator:
    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.baseline_post_attempt: dict[str, Any] | None = None
        self.best_checkpoint_report: dict[str, Any] | None = None
        self.qualification_template_report: dict[str, Any] | None = None
        self.artifact_qualification_reports: list[dict[str, Any]] = []

    def run(self, config: RunConfig) -> list[IterationResult]:
        harness_dir = ensure_harness_dirs(config.workspace)
        registry = load_registry(config.workspace)
        save_registry(config.workspace, registry)
        task_state = TaskStateManager(config.workspace, config.task_state)
        task_state.capture_initial()

        self.run_id = config.run_id or make_run_id()
        self.run_dir = run_dir(config.workspace, self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.qualification_template_report = None
        self.artifact_qualification_reports = []
        if config.qualify_artifacts:
            self.qualification_template_report = capture_clean_workspace_template(
                config.workspace,
                self.run_dir / "qualification" / "workspace_template",
            )
        self._write_run_config(config)
        best_checkpoint = BestCheckpointManager(config.workspace, self.run_dir)
        self.best_checkpoint_report = {
            "enabled": bool(config.optimization_mode and config.restore_best),
            "restored": False,
            "captures": [],
        }
        if config.optimization_mode and not config.restore_best:
            self.best_checkpoint_report["reason"] = "disabled_by_config"
        self.baseline_post_attempt = self._run_baseline_post_attempt(config)
        baseline_score = (
            self.baseline_post_attempt.get("score")
            if isinstance(self.baseline_post_attempt, dict)
            else None
        )
        best_score = baseline_score if isinstance(baseline_score, (int, float)) else None
        if self._should_capture_baseline_checkpoint(config, baseline_score):
            capture = best_checkpoint.capture(
                iteration=0,
                score=float(baseline_score),
                reason="baseline_post_attempt_passed",
            )
            self._record_best_checkpoint_capture(capture)

        runner = self.runner or make_runner(config)
        results: list[IterationResult] = []

        try:
            for iteration in range(1, config.iterations + 1):
                task_state.prepare_iteration(iteration)
                registry = load_registry(config.workspace)
                registry = sync_registry_from_files(config.workspace, registry, iteration=None)
                activation = activate_generated_harness(config.workspace, registry)
                current_iteration_dir = iteration_dir(config.workspace, self.run_id, iteration)
                current_iteration_dir.mkdir(parents=True, exist_ok=True)
                result_path = current_iteration_dir / "result.json"
                create_initial_result(
                    path=result_path,
                    run_id=self.run_id,
                    iteration=iteration,
                    task_path=config.task_path,
                    registry=registry,
                )
                context = IterationContext(
                    run_id=self.run_id,
                    iteration=iteration,
                    workspace=config.workspace,
                    harness_dir=harness_dir,
                    run_dir=self.run_dir,
                    iteration_dir=current_iteration_dir,
                    result_path=result_path,
                    registry=registry,
                    task_text=config.task_text,
                    task_path=config.task_path,
                    artifact_context=render_artifact_context(registry, config.workspace),
                    attempt_timeout_seconds=config.attempt_timeout_for(iteration),
                )
                write_json(current_iteration_dir / "activation.json", activation)

                attempt = runner.start_attempt(config, context, render_attempt_prompt(config, context))
                self._record_phase(result_path, "attempt", attempt)
                attempt_data = read_json(result_path)
                attempt_data = self._record_post_attempt(config, context, attempt_data, baseline_score, best_score)
                attempt_data = self._record_harness_usage(result_path, activation, registry, attempt)
                score = extract_score(attempt_data, config.score_key)
                if self._should_capture_iteration_checkpoint(config, attempt_data, score, best_score):
                    capture = best_checkpoint.capture(
                        iteration=iteration,
                        score=float(score),
                        reason="post_attempt_best_score",
                    )
                    self._record_best_checkpoint_capture(capture)
                    attempt_data = update_result(result_path, {"optimization": {"best_checkpoint": capture}})
                    best_score = score
                elif score is not None and (best_score is None or self._is_better(score, best_score, config)):
                    best_score = score
                if config.attempt_only:
                    solved = is_solved_and_verified(attempt_data) or reached_stop_score(attempt_data, config)
                    results.append(self._iteration_result(iteration, result_path, attempt, None, None, solved))
                    if solved:
                        break
                    continue
                if reached_stop_score(attempt_data, config):
                    results.append(self._iteration_result(iteration, result_path, attempt, None, None, True))
                    break
                if is_solved_and_verified(attempt_data) and config.stop_score is None:
                    if config.build_after_solve:
                        reflection, build = self._reflect_and_build(config, context, runner, attempt, attempt_data)
                        registry, post_build_activation = self._finalize_artifacts_after_build(
                            config,
                            context,
                            runner,
                            build,
                        )
                        update_result(
                            result_path,
                            {
                                "status": "solved",
                                "verified": True,
                                "primary_task_solved_before_build": True,
                                "registry_artifact_count": len(registry.artifacts),
                                "registry_path": str(config.workspace / ".harnessgym" / "registry.json"),
                                "harness_quality_gate": post_build_activation.get("quality_gate", {}),
                            },
                        )
                        results.append(
                            self._iteration_result(iteration, result_path, attempt, reflection, build, True)
                        )
                    else:
                        results.append(self._iteration_result(iteration, result_path, attempt, None, None, True))
                    break

                reflection, build = self._reflect_and_build(config, context, runner, attempt, attempt_data)

                registry, post_build_activation = self._finalize_artifacts_after_build(
                    config,
                    context,
                    runner,
                    build,
                )
                final_data = update_result(
                    result_path,
                    {
                        "registry_artifact_count": len(registry.artifacts),
                        "registry_path": str(config.workspace / ".harnessgym" / "registry.json"),
                        "harness_quality_gate": post_build_activation.get("quality_gate", {}),
                    },
                )
                final_data = self._record_harness_usage(result_path, activation, registry, attempt)
                if (
                    final_data.get("status") in (None, "running", "blocked", "incomplete")
                    and build.status == "completed"
                    and not is_solved_and_verified(final_data)
                ):
                    final_data = update_result(result_path, {"status": "tooling_built"})

                solved = is_solved_and_verified(final_data)
                score_reached = reached_stop_score(final_data, config)
                results.append(self._iteration_result(iteration, result_path, attempt, reflection, build, solved))
                if (solved and config.stop_score is None) or score_reached:
                    break
        finally:
            runner.close()

        if config.optimization_mode and config.restore_best:
            self._restore_best_checkpoint_if_available(config, best_checkpoint, results)

        self._write_run_summary(results, config)
        return results

    def _reflect_and_build(
        self,
        config: RunConfig,
        context: IterationContext,
        runner: Runner,
        attempt: RunnerResult,
        attempt_data: dict,
    ) -> tuple[RunnerResult, RunnerResult]:
        reflection_prompt = render_reflection_prompt(config, context, attempt_data)
        reflection = runner.reflect(config, context, reflection_prompt, attempt.session_id)
        self._record_phase(context.result_path, "reflection", reflection)

        reflected_data = read_json(context.result_path)
        selected = extract_selected_improvement(
            reflected_data,
            fallback=self._read_optional_text(reflection.stdout_path),
        )
        build_prompt = render_build_prompt(config, context, selected)
        build_session_id = reflection.session_id or attempt.session_id
        build = runner.build_tooling(config, context, build_prompt, build_session_id)
        self._record_phase(context.result_path, "build", build)
        return reflection, build

    def _finalize_artifacts_after_build(
        self,
        config: RunConfig,
        context: IterationContext,
        runner: Runner,
        build: RunnerResult,
    ) -> tuple[Any, dict[str, Any]]:
        registry = sync_registry_from_files(config.workspace, load_registry(config.workspace), context.iteration)
        if config.qualify_artifacts:
            registry = self._qualify_and_repair_artifacts(config, context, runner, build, registry)
        post_build_activation = activate_generated_harness(config.workspace, registry)
        write_json(context.iteration_dir / "post_build_activation.json", post_build_activation)
        return registry, post_build_activation

    def _qualify_and_repair_artifacts(
        self,
        config: RunConfig,
        context: IterationContext,
        runner: Runner,
        build: RunnerResult,
        registry: Any,
    ) -> Any:
        if self.run_dir is None or self.qualification_template_report is None:
            return registry
        session_id = build.session_id
        report: dict[str, Any] | None = None
        for attempt_index in range(0, config.artifact_repair_attempts + 1):
            qualification_dir = context.iteration_dir / "qualification" / f"attempt_{attempt_index + 1}"
            report = qualify_generated_harness(
                source_workspace=config.workspace,
                template_dir=Path(str(self.qualification_template_report["template_path"])),
                qualification_dir=qualification_dir,
                iteration=context.iteration,
                registry=registry,
            )
            report_path = qualification_dir / "qualification.json"
            report["report_path"] = str(report_path)
            self.artifact_qualification_reports.append(_qualification_summary(report))
            update_result(context.result_path, {"artifact_qualification": report})
            if report.get("status") == "passed":
                qualified = mark_artifacts_qualified(
                    registry,
                    paths=[artifact.path for artifact in registry.artifacts],
                    report_path=str(report_path),
                    iteration=context.iteration,
                )
                report["qualified_artifacts"] = qualified
                write_json(report_path, report)
                save_registry(config.workspace, registry)
                update_result(context.result_path, {"artifact_qualification": report})
                return registry
            if attempt_index >= config.artifact_repair_attempts or not session_id:
                break
            repair_dir = context.iteration_dir / f"repair_{attempt_index + 1}"
            repair_dir.mkdir(parents=True, exist_ok=True)
            repair_context = replace(context, iteration_dir=repair_dir)
            repair_prompt = render_repair_prompt(config, context, report, attempt_index + 1)
            repair = runner.build_tooling(config, repair_context, repair_prompt, session_id)
            session_id = repair.session_id or session_id
            self._record_phase(context.result_path, f"repair_{attempt_index + 1}", repair)
            registry = sync_registry_from_files(config.workspace, load_registry(config.workspace), context.iteration)

        if report is None:
            return registry
        failed_paths = self._qualification_failed_paths(report, registry, context.iteration)
        quarantined = quarantine_artifacts(
            registry,
            paths=failed_paths,
            reason=str(report.get("quality_gate", {}).get("warnings") or report.get("quality_gate", {}).get("status")),
            report_path=str(report.get("report_path") or ""),
            iteration=context.iteration,
        )
        report["quarantined_artifacts"] = quarantined
        report_path = Path(str(report.get("report_path") or context.iteration_dir / "qualification.json"))
        write_json(report_path, report)
        save_registry(config.workspace, registry)
        update_result(context.result_path, {"artifact_qualification": report})
        return registry

    def _qualification_failed_paths(self, report: dict[str, Any], registry: Any, iteration: int) -> list[str]:
        paths = [str(path) for path in report.get("failed_artifacts", []) if path]
        paths.extend(str(artifact.path) for artifact in registry.artifacts if artifact.iteration == iteration)
        return sorted(dict.fromkeys(paths))

    def _write_run_config(self, config: RunConfig) -> None:
        assert self.run_dir is not None
        write_json(
            self.run_dir / "run_config.json",
            {
                "run_id": self.run_id,
                "workspace": str(config.workspace),
                "task_path": str(config.task_path) if config.task_path else None,
                "iterations": config.iterations,
                "attempt_timeout_seconds": config.attempt_timeout_seconds,
                "attempt_timeout_schedule_seconds": list(config.attempt_timeout_schedule_seconds),
                "build_timeout_seconds": config.build_timeout_seconds,
                "reflection_timeout_seconds": config.reflection_timeout_seconds,
                "runner": config.runner,
                "codex_bin": config.codex_bin,
                "claude_bin": config.claude_bin,
                "claude_model": config.claude_model,
                "claude_permission_mode": config.claude_permission_mode,
                "claude_max_budget_usd": config.claude_max_budget_usd,
                "claude_extra_args": list(config.claude_extra_args),
                "build_after_solve": config.build_after_solve,
                "stop_score": config.stop_score,
                "score_key": config.score_key,
                "higher_is_better": config.higher_is_better,
                "task_state": config.task_state,
                "attempt_only": config.attempt_only,
                "harness_depth": config.harness_depth,
                "post_attempt_command": config.post_attempt_command,
                "post_attempt_score_key": config.post_attempt_score_key or config.score_key,
                "post_attempt_timeout_seconds": config.post_attempt_timeout_seconds,
                "optimization_mode": config.optimization_mode,
                "restore_best": config.restore_best,
                "qualify_artifacts": config.qualify_artifacts,
                "artifact_repair_attempts": config.artifact_repair_attempts,
            },
        )

    def _write_run_summary(self, results: list[IterationResult], config: RunConfig) -> None:
        if self.run_dir is None:
            return
        write_json(
            self.run_dir / "summary.json",
            {
                "run_id": self.run_id,
                "iterations": [result.to_dict() for result in results],
                "final_status": results[-1].status if results else "not_started",
                "solved": bool(results and results[-1].solved),
                "best_score": self._best_score(results, config),
                "baseline_post_attempt": self.baseline_post_attempt,
                "optimization": self._optimization_summary(results, config),
                "best_checkpoint": self.best_checkpoint_report,
                "artifact_qualification": {
                    "template": self.qualification_template_report,
                    "reports": self.artifact_qualification_reports,
                },
                "harness_usage": self._harness_usage_summary(results),
                "harness_quality_gates": self._harness_quality_gate_summary(results),
            },
        )

    def _best_score(self, results: list[IterationResult], config: RunConfig) -> float | None:
        scores = []
        if isinstance(self.baseline_post_attempt, dict):
            baseline_score = self.baseline_post_attempt.get("score")
            if isinstance(baseline_score, (int, float)):
                scores.append(float(baseline_score))
        for result in results:
            data = read_json(Path(result.result_path))
            score = extract_score(data, config.score_key)
            if score is not None:
                scores.append(score)
        if not scores:
            return None
        return max(scores) if config.higher_is_better else min(scores)

    def _run_baseline_post_attempt(self, config: RunConfig) -> dict[str, Any] | None:
        if not (config.optimization_mode and config.post_attempt_command and self.run_dir is not None):
            return None
        score_key = config.post_attempt_score_key or config.score_key
        return run_post_command(
            command=config.post_attempt_command,
            cwd=config.workspace,
            log_dir=self.run_dir / "baseline",
            timeout_seconds=config.post_attempt_timeout_seconds,
            score_key=score_key,
            prefix="baseline",
        )

    def _record_post_attempt(
        self,
        config: RunConfig,
        context: IterationContext,
        attempt_data: dict,
        baseline_score: float | None,
        best_score: float | None,
    ) -> dict:
        if not config.post_attempt_command:
            return attempt_data
        score_key = config.post_attempt_score_key or config.score_key
        post_result = run_post_command(
            command=config.post_attempt_command,
            cwd=context.workspace,
            log_dir=context.iteration_dir,
            timeout_seconds=config.post_attempt_timeout_seconds,
            score_key=score_key,
            prefix="post_attempt",
        )
        post_passed = post_command_passed(post_result)
        score = post_result.get("score")
        updates: dict[str, Any] = {
            "post_attempt": post_result,
            "verification": {
                "status": "passed" if post_passed else post_result["status"],
                "command": config.post_attempt_command,
                "return_code": post_result.get("return_code"),
                "timed_out": post_result.get("timed_out"),
                "stdout_path": post_result.get("stdout_path"),
                "stderr_path": post_result.get("stderr_path"),
            },
        }
        if score is not None:
            updates["verification"][config.score_key] = score
            updates["verification"]["score"] = score
            updates["metrics"] = {
                config.score_key: score,
                score_key: score,
                "score": score,
            }
        if config.optimization_mode:
            updates["optimization"] = self._iteration_optimization_state(
                score=score if isinstance(score, (int, float)) else None,
                baseline_score=baseline_score,
                previous_best_score=best_score,
                config=config,
            )
            if score is not None and self._score_reaches_target(float(score), config):
                updates["status"] = "solved"
                updates["verified"] = post_passed
            elif post_passed:
                improved = self._is_better_than_baseline(float(score), baseline_score, config) if score is not None else False
                updates["status"] = "improved" if improved else "verified"
                updates["verified"] = True
        elif post_passed:
            updates["status"] = "solved"
            updates["verified"] = True
        return update_result(context.result_path, updates)

    def _record_harness_usage(
        self,
        result_path: Path,
        activation: dict[str, Any],
        registry: Any,
        runner_result: RunnerResult | None = None,
    ) -> dict:
        data = read_json(result_path)
        active_tools: set[str] = set()
        active_mcps: list[str] = []
        for server in activation.get("mcp_servers", []):
            if server.get("active") is False:
                continue
            name = server.get("name")
            if name:
                active_mcps.append(str(name))
            active_tools.update(str(tool) for tool in server.get("enabled_tools", []) if tool)
            smoke = server.get("smoke")
            if isinstance(smoke, dict):
                active_tools.update(str(tool) for tool in smoke.get("tools", []) if tool)
        active_skills = [
            str(skill.get("artifact_path"))
            for skill in activation.get("skills", [])
            if skill.get("artifact_path")
        ]
        used_tools = _string_list(data.get("used_harness_tools") or data.get("used_tools"))
        used_artifacts = _string_list(data.get("used_harness_artifacts") or data.get("used_artifacts"))
        registry_paths = [
            str(artifact.path)
            for artifact in getattr(registry, "artifacts", [])
            if getattr(artifact, "path", None)
        ]
        inferred = _infer_harness_usage_from_outputs(
            runner_result,
            tool_names=active_tools,
            artifact_paths=set(registry_paths + active_skills),
        )
        combined_tools = sorted(dict.fromkeys([*used_tools, *inferred["tools"]]))
        combined_artifacts = sorted(dict.fromkeys([*used_artifacts, *inferred["artifacts"]]))
        mcp_telemetry = summarize_mcp_call_events(_workspace_from_result_path(result_path))
        telemetry_tools = [
            tool for tool in mcp_telemetry.get("called_tools", []) if isinstance(tool, str)
        ]
        combined_tools = sorted(dict.fromkeys([*combined_tools, *telemetry_tools]))
        usage = {
            "registry_artifact_count": len(registry.artifacts),
            "active_skill_count": len(active_skills),
            "active_mcp_count": len(active_mcps),
            "active_tool_count": len(active_tools),
            "active_skills": sorted(active_skills),
            "active_mcp_servers": sorted(active_mcps),
            "active_tools": sorted(active_tools),
            "used_harness_tools": combined_tools,
            "used_harness_artifacts": combined_artifacts,
            "declared_harness_tools": used_tools,
            "declared_harness_artifacts": used_artifacts,
            "inferred_harness_tools": inferred["tools"],
            "inferred_harness_artifacts": inferred["artifacts"],
            "inference_sources": inferred["sources"],
            "used_active_tools": sorted(set(combined_tools) & active_tools),
            "mcp_call_count": mcp_telemetry.get("count", 0),
            "mcp_successful_call_count": mcp_telemetry.get("successful_count", 0),
            "mcp_called_tools": mcp_telemetry.get("called_tools", []),
            "mcp_call_status_counts": mcp_telemetry.get("status_counts", {}),
            "declared_tool_use_count": len(used_tools),
            "declared_artifact_use_count": len(used_artifacts),
            "inferred_tool_use_count": len(inferred["tools"]),
            "inferred_artifact_use_count": len(inferred["artifacts"]),
        }
        return update_result(result_path, {"harness_usage": usage})

    def _optimization_summary(self, results: list[IterationResult], config: RunConfig) -> dict[str, Any]:
        baseline_score = (
            self.baseline_post_attempt.get("score")
            if isinstance(self.baseline_post_attempt, dict)
            else None
        )
        best_score = self._best_score(results, config)
        summary = {
            "enabled": config.optimization_mode,
            "score_key": config.score_key,
            "higher_is_better": config.higher_is_better,
            "baseline_score": baseline_score,
            "best_score": best_score,
            "improved": False,
            "restore_best": config.restore_best,
            "best_checkpoint": self.best_checkpoint_report,
        }
        if isinstance(baseline_score, (int, float)) and isinstance(best_score, (int, float)):
            summary["improved"] = self._is_better(best_score, baseline_score, config)
            summary["absolute_delta_from_baseline"] = best_score - baseline_score
            if baseline_score:
                if config.higher_is_better:
                    summary["relative_improvement"] = (best_score / baseline_score) - 1.0
                else:
                    summary["relative_reduction"] = 1.0 - (best_score / baseline_score)
        return summary

    def _harness_usage_summary(self, results: list[IterationResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in results:
            data = read_json(Path(result.result_path))
            usage = data.get("harness_usage")
            if isinstance(usage, dict):
                rows.append(
                    {
                        "iteration": result.iteration,
                        "active_skill_count": usage.get("active_skill_count"),
                        "active_mcp_count": usage.get("active_mcp_count"),
                        "active_tool_count": usage.get("active_tool_count"),
                        "declared_tool_use_count": usage.get("declared_tool_use_count"),
                        "inferred_tool_use_count": usage.get("inferred_tool_use_count"),
                        "used_active_tools": usage.get("used_active_tools"),
                    }
                )
        return rows

    def _harness_quality_gate_summary(self, results: list[IterationResult]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for result in results:
            data = read_json(Path(result.result_path))
            gate = data.get("harness_quality_gate")
            if isinstance(gate, dict):
                rows.append(
                    {
                        "iteration": result.iteration,
                        "status": gate.get("status"),
                        "active_mcp_count": gate.get("active_mcp_count"),
                        "inactive_mcp_count": gate.get("inactive_mcp_count"),
                        "active_tool_count": gate.get("active_tool_count"),
                    }
                )
        return rows

    def _iteration_optimization_state(
        self,
        *,
        score: float | None,
        baseline_score: float | None,
        previous_best_score: float | None,
        config: RunConfig,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "mode": "optimization",
            "score_key": config.score_key,
            "higher_is_better": config.higher_is_better,
            "baseline_score": baseline_score,
            "previous_best_score": previous_best_score,
            "current_score": score,
        }
        if score is not None:
            state["target_reached"] = self._score_reaches_target(score, config)
            if baseline_score is not None:
                state["improved_from_baseline"] = self._is_better(score, baseline_score, config)
                state["delta_from_baseline"] = score - baseline_score
                if baseline_score:
                    if config.higher_is_better:
                        state["relative_improvement"] = (score / baseline_score) - 1.0
                    else:
                        state["relative_reduction"] = 1.0 - (score / baseline_score)
            if previous_best_score is not None:
                state["improved_from_previous_best"] = self._is_better(score, previous_best_score, config)
        return state

    def _score_reaches_target(self, score: float, config: RunConfig) -> bool:
        if config.stop_score is None:
            return False
        if config.higher_is_better:
            return score >= config.stop_score
        return score <= config.stop_score

    def _is_better_than_baseline(self, score: float, baseline_score: float | None, config: RunConfig) -> bool:
        return baseline_score is not None and self._is_better(score, baseline_score, config)

    def _is_better(self, score: float, reference: float, config: RunConfig) -> bool:
        if config.higher_is_better:
            return score > reference
        return score < reference

    def _should_capture_baseline_checkpoint(self, config: RunConfig, baseline_score: Any) -> bool:
        return (
            config.optimization_mode
            and config.restore_best
            and isinstance(self.baseline_post_attempt, dict)
            and post_command_passed(self.baseline_post_attempt)
            and isinstance(baseline_score, (int, float))
        )

    def _should_capture_iteration_checkpoint(
        self,
        config: RunConfig,
        result_data: dict[str, Any],
        score: float | None,
        best_score: float | None,
    ) -> bool:
        if not (config.optimization_mode and config.restore_best and isinstance(score, (int, float))):
            return False
        if not self._checkpointable_result(result_data):
            return False
        return best_score is None or self._is_better(score, best_score, config)

    def _checkpointable_result(self, result_data: dict[str, Any]) -> bool:
        post_attempt = result_data.get("post_attempt")
        if isinstance(post_attempt, dict):
            return post_command_passed(post_attempt)
        return is_solved_and_verified(result_data)

    def _record_best_checkpoint_capture(self, capture: dict[str, Any]) -> None:
        if self.best_checkpoint_report is None:
            return
        captures = self.best_checkpoint_report.setdefault("captures", [])
        if isinstance(captures, list):
            captures.append(capture)
        self.best_checkpoint_report["latest"] = capture

    def _restore_best_checkpoint_if_available(
        self,
        config: RunConfig,
        best_checkpoint: BestCheckpointManager,
        results: list[IterationResult],
    ) -> None:
        if self.best_checkpoint_report is None:
            return
        if not best_checkpoint.has_checkpoint():
            self.best_checkpoint_report["reason"] = "no_checkpoint_captured"
            return
        restore_report = best_checkpoint.restore()
        manifest = read_json(best_checkpoint.manifest_path)
        restore_report["iteration"] = manifest.get("iteration")
        restore_report["score"] = manifest.get("score")
        restore_report["reason"] = manifest.get("reason")
        if config.post_attempt_command and self.run_dir is not None:
            score_key = config.post_attempt_score_key or config.score_key
            post_restore = run_post_command(
                command=config.post_attempt_command,
                cwd=config.workspace,
                log_dir=self.run_dir / "checkpoints",
                timeout_seconds=config.post_attempt_timeout_seconds,
                score_key=score_key,
                prefix="restored_best",
            )
            restore_report["post_restore"] = post_restore
            restore_report["post_restore_passed"] = post_command_passed(post_restore)
        self.best_checkpoint_report["restored"] = True
        self.best_checkpoint_report["restore"] = restore_report
        if results:
            update_result(
                Path(results[-1].result_path),
                {"optimization": {"restored_best_checkpoint": restore_report}},
            )

    def _record_phase(self, result_path: Path, phase: str, result: RunnerResult) -> None:
        data = update_result(result_path, {"phases": {phase: result.to_dict()}})
        if data.get("status") == "running" and phase == "attempt":
            fallback_status = {
                "timeout": "incomplete",
                "failed": "failed",
                "completed": "incomplete",
            }.get(result.status, result.status)
            update_result(result_path, {"status": fallback_status})

    def _iteration_result(
        self,
        iteration: int,
        result_path: Path,
        attempt: RunnerResult | None,
        reflection: RunnerResult | None,
        build: RunnerResult | None,
        solved: bool,
    ) -> IterationResult:
        data = read_json(result_path)
        return IterationResult(
            iteration=iteration,
            status=str(data.get("status", "unknown")),
            result_path=str(result_path),
            attempt=attempt,
            reflection=reflection,
            build=build,
            solved=solved,
        )

    def _read_optional_text(self, path: str | None) -> str:
        if not path:
            return ""
        candidate = Path(path)
        if not candidate.exists():
            return ""
        return candidate.read_text(encoding="utf-8")


def is_solved_and_verified(result_data: dict) -> bool:
    if result_data.get("status") != "solved":
        return False
    if result_data.get("verified") is True:
        return True
    verification = result_data.get("verification")
    return isinstance(verification, dict) and verification.get("status") == "passed"


def extract_score(result_data: dict, score_key: str) -> float | None:
    return extract_score_from_payload(result_data, score_key)


def reached_stop_score(result_data: dict, config: RunConfig) -> bool:
    if config.stop_score is None:
        return False
    score = extract_score(result_data, config.score_key)
    if score is None:
        return False
    if config.higher_is_better:
        return score >= config.stop_score
    return score <= config.stop_score


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, dict):
            path = item.get("path")
            name = item.get("name") or item.get("id")
            if path:
                strings.append(str(path))
            elif name:
                strings.append(str(name))
    return sorted(dict.fromkeys(strings))


def _workspace_from_result_path(result_path: Path) -> Path:
    for parent in result_path.parents:
        if parent.name == ".harnessgym":
            return parent.parent
    return result_path.parent


def _infer_harness_usage_from_outputs(
    runner_result: RunnerResult | None,
    *,
    tool_names: set[str],
    artifact_paths: set[str],
) -> dict[str, list[str]]:
    if runner_result is None:
        return {"tools": [], "artifacts": [], "sources": []}
    sources: list[str] = []
    chunks: list[str] = []
    for raw_path in (runner_result.stdout_path, runner_result.stderr_path, runner_result.transcript_path):
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text:
            sources.append(str(path))
            chunks.append(text)
    if not chunks:
        return {"tools": [], "artifacts": [], "sources": []}
    haystack = "\n".join(chunks).lower()
    tools = sorted(
        tool
        for tool in tool_names
        if tool and _contains_usage_token(haystack, tool)
    )
    artifacts = sorted(
        artifact
        for artifact in artifact_paths
        if artifact and _contains_artifact_reference(haystack, artifact)
    )
    return {"tools": tools, "artifacts": artifacts, "sources": sources}


def _contains_artifact_reference(haystack: str, path: str) -> bool:
    haystack = haystack.lower()
    normalized = path.lower()
    if normalized in haystack:
        return True
    basename = normalized.rsplit("/", 1)[-1]
    return bool(basename and basename != normalized and basename in haystack)


def _contains_usage_token(haystack: str, token: str) -> bool:
    normalized = token.lower()
    # Match MCP tool-call patterns or word-boundary occurrences rather than
    # bare substring presence to avoid false positives like "I will run the
    # tests" matching tool "run".
    # Patterns: mcp__server__tool, "tools/call" with "name":"tool", or the
    # tool name as a whole word near "tool" context.
    if f"mcp__{normalized}" in haystack:
        return True
    if f"__{normalized}" in haystack and "mcp__" in haystack:
        return True
    if f'"{normalized}"' in haystack and "tools/call" in haystack:
        return True
    basename = normalized.rsplit("/", 1)[-1]
    candidates = [normalized] if basename == normalized else [normalized, basename]
    for candidate in candidates:
        if not candidate:
            continue
        if f"mcp__{candidate}" in haystack:
            return True
        if f"__{candidate}" in haystack and "mcp__" in haystack:
            return True
        if f'"{candidate}"' in haystack and "tools/call" in haystack:
            return True
        # Word-boundary match near "tool" keyword context (e.g. "MCP tool run_verifier").
        for match in re.finditer(rf"\b{re.escape(candidate)}\b", haystack):
            window = haystack[max(0, match.start() - 30) : match.end() + 30]
            if "tool" in window:
                return True
    return False


def _qualification_summary(report: dict[str, Any]) -> dict[str, Any]:
    gate = report.get("quality_gate") if isinstance(report.get("quality_gate"), dict) else {}
    return {
        "iteration": report.get("iteration"),
        "status": report.get("status"),
        "report_path": report.get("report_path"),
        "fresh_workspace": report.get("fresh_workspace"),
        "active_mcp_count": gate.get("active_mcp_count"),
        "active_tool_count": gate.get("active_tool_count"),
        "inactive_mcp_count": gate.get("inactive_mcp_count"),
        "failed_artifacts": report.get("failed_artifacts"),
    }
