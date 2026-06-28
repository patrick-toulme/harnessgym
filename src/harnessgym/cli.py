from __future__ import annotations

import argparse
import sys

from .compare import CompareConfig, run_compare
from .config import HARNESS_DEPTH_CHOICES, RUNNER_CHOICES, RunConfig
from .orchestrator import Orchestrator
from .task_state import TASK_STATE_CHOICES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harnessgym", description="Iterative Codex harness improvement.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an iterative HarnessGym loop.")
    task_group = run.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Path to a task markdown/text file.")
    task_group.add_argument("--task-text", help="Inline task text.")
    run.add_argument("--workspace", default=".", help="Workspace directory for the task.")
    run.add_argument("--iterations", type=int, default=3, help="Maximum iteration count.")
    run.add_argument("--attempt-timeout", default="45m", help="Attempt phase timeout, for example 45m.")
    run.add_argument(
        "--attempt-timeouts",
        default=None,
        help="Comma-separated per-iteration attempt timeouts, for example 10s,2m. Overrides --attempt-timeout by iteration.",
    )
    run.add_argument("--build-timeout", default="20m", help="Build phase timeout, for example 20m.")
    run.add_argument(
        "--reflection-timeout",
        default=None,
        help="Reflection phase timeout. Defaults to --build-timeout.",
    )
    run.add_argument("--runner", choices=RUNNER_CHOICES, default="exec", help="Runner backend.")
    run.add_argument("--codex-bin", default="codex", help="Codex executable for exec and tui-goal runners.")
    run.add_argument("--claude-bin", default="claude", help="Claude Code executable for the claude runner.")
    run.add_argument("--claude-model", default=None, help="Optional Claude model or alias, for example sonnet or opus.")
    run.add_argument(
        "--claude-permission-mode",
        default="bypassPermissions",
        help="Claude Code permission mode for the claude runner. Defaults to bypassPermissions for autonomous runs.",
    )
    run.add_argument(
        "--claude-max-budget-usd",
        type=float,
        default=None,
        help="Optional Claude Code print-mode spend cap per phase.",
    )
    run.add_argument(
        "--claude-extra-arg",
        action="append",
        default=[],
        help="Extra argument to pass to Claude Code. Repeat for multiple arguments.",
    )
    run.add_argument("--run-id", default=None, help="Optional deterministic run id.")
    run.add_argument(
        "--build-after-solve",
        action="store_true",
        help="After a verified solve, still run reflection/build to generate reusable harness artifacts for future runs.",
    )
    run.add_argument("--stop-score", type=float, default=None, help="Stop early when result metric reaches this score.")
    run.add_argument("--score-key", default="score", help="Metric key to read from result.json metrics/objective.")
    run.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger score values as better. Default assumes lower is better.",
    )
    run.add_argument(
        "--task-state",
        choices=TASK_STATE_CHOICES,
        default="continue",
        help="continue compounds task edits across iterations; reset restores task files before each new iteration while preserving .harnessgym artifacts.",
    )
    run.add_argument(
        "--harness-depth",
        choices=HARNESS_DEPTH_CHOICES,
        default="deep",
        help="deep steers reflection/build toward executable domain instrumentation and richer MCP tooling; standard builds smaller focused artifacts.",
    )
    run.add_argument(
        "--post-attempt-command",
        help="Optional JSON-emitting command to run after every attempt, even timed-out attempts, for independent scoring/verification.",
    )
    run.add_argument(
        "--post-attempt-score-key",
        default=None,
        help="Metric key to read from --post-attempt-command JSON output. Defaults to --score-key.",
    )
    run.add_argument("--post-attempt-timeout", default="2m", help="Timeout for --post-attempt-command.")
    run.add_argument(
        "--optimization-mode",
        action="store_true",
        help="Treat post-attempt scores as an open-ended optimization objective and report improvement even when --stop-score is not reached.",
    )
    run.add_argument(
        "--restore-best",
        dest="restore_best",
        action="store_true",
        default=True,
        help="In optimization mode, restore the best independently scored task workspace after the run. Enabled by default.",
    )
    run.add_argument(
        "--no-restore-best",
        dest="restore_best",
        action="store_false",
        help="Leave the workspace at the final attempted state instead of restoring the best optimization checkpoint.",
    )
    run.add_argument(
        "--qualify-artifacts",
        dest="qualify_artifacts",
        action="store_true",
        default=True,
        help="After each build, validate generated harness artifacts in a fresh copied workspace. Enabled by default.",
    )
    run.add_argument(
        "--no-qualify-artifacts",
        dest="qualify_artifacts",
        action="store_false",
        help="Skip fresh-workspace qualification and quarantine checks for generated artifacts.",
    )
    run.add_argument(
        "--artifact-repair-attempts",
        type=int,
        default=1,
        help="Number of same-session repair build attempts to try after failed fresh artifact qualification.",
    )
    run.set_defaults(func=run_command)

    compare = subparsers.add_parser("compare", help="Replay plain vs harnessed attempts across copied workspaces.")
    compare_task_group = compare.add_mutually_exclusive_group(required=True)
    compare_task_group.add_argument("--task", help="Path to a task file, preferably relative to --workspace-template.")
    compare_task_group.add_argument("--task-text", help="Inline task text.")
    compare.add_argument("--workspace-template", required=True, help="Clean workspace template to copy for each trial.")
    compare.add_argument("--artifact-source", help="Workspace or .harnessgym directory containing generated artifacts.")
    compare.add_argument("--output-dir", required=True, help="Directory where compare workspaces and report are written.")
    compare.add_argument("--trials", type=int, default=1, help="Number of plain and harnessed trials to run.")
    compare.add_argument("--iterations", type=int, default=1, help="Iterations per replay trial. Defaults to one attempt.")
    compare.add_argument("--attempt-timeout", default="5m", help="Attempt timeout per replay iteration.")
    compare.add_argument(
        "--attempt-timeouts",
        default=None,
        help="Comma-separated per-iteration attempt timeouts for replay trials.",
    )
    compare.add_argument("--build-timeout", default="1s", help="Kept for config compatibility; compare runs are attempt-only.")
    compare.add_argument(
        "--reflection-timeout",
        default="1s",
        help="Kept for config compatibility; compare runs are attempt-only.",
    )
    compare.add_argument("--runner", choices=RUNNER_CHOICES, default="exec", help="Runner backend.")
    compare.add_argument("--codex-bin", default="codex", help="Codex executable for exec and tui-goal runners.")
    compare.add_argument("--claude-bin", default="claude", help="Claude Code executable for the claude runner.")
    compare.add_argument("--claude-model", default=None, help="Optional Claude model or alias, for example sonnet or opus.")
    compare.add_argument(
        "--claude-permission-mode",
        default="bypassPermissions",
        help="Claude Code permission mode for the claude runner. Defaults to bypassPermissions for autonomous runs.",
    )
    compare.add_argument(
        "--claude-max-budget-usd",
        type=float,
        default=None,
        help="Optional Claude Code print-mode spend cap per phase.",
    )
    compare.add_argument(
        "--claude-extra-arg",
        action="append",
        default=[],
        help="Extra argument to pass to Claude Code. Repeat for multiple arguments.",
    )
    compare.add_argument("--stop-score", type=float, default=None, help="Stop a replay early when result metric reaches this score.")
    compare.add_argument("--score-key", default="score", help="Metric key to read from result.json metrics/objective.")
    compare.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger score values as better. Default assumes lower is better.",
    )
    compare.add_argument(
        "--task-state",
        choices=TASK_STATE_CHOICES,
        default="reset",
        help="Task state mode inside each replay trial. Default reset keeps attempts comparable.",
    )
    compare.add_argument(
        "--build-after-solve",
        action="store_true",
        help="Accepted for config compatibility; compare replay still measures attempt-only runs.",
    )
    compare.add_argument(
        "--post-command",
        help="Optional command to run in each trial workspace after the attempt, for example 'python3 benchmark.py --json --mode final'.",
    )
    compare.add_argument("--post-score-key", default="score", help="Metric key to read from --post-command JSON output.")
    compare.add_argument("--post-timeout", default="2m", help="Timeout for --post-command.")
    compare.add_argument(
        "--require-active-harness",
        dest="require_active_harness",
        action="store_true",
        default=True,
        help="Mark harnessed trials invalid unless copied artifacts activate at least one generated MCP tool. Enabled by default.",
    )
    compare.add_argument(
        "--no-require-active-harness",
        dest="require_active_harness",
        action="store_false",
        help="Allow harnessed replay trials even when no generated MCP tools activate.",
    )
    compare.add_argument(
        "--require-harness-tool-use",
        dest="require_harness_tool_use",
        action="store_true",
        default=False,
        help="Mark harnessed trials invalid unless at least one generated MCP tool call is recorded.",
    )
    compare.add_argument(
        "--no-require-harness-tool-use",
        dest="require_harness_tool_use",
        action="store_false",
        help="Do not require recorded generated MCP tool calls in harnessed trials.",
    )
    compare.add_argument("--overwrite", action="store_true", help="Replace existing trial directories under --output-dir.")
    compare.set_defaults(func=compare_command)
    return parser


def run_command(args: argparse.Namespace) -> int:
    config = RunConfig.from_values(
        workspace=args.workspace,
        task=args.task,
        task_text=args.task_text,
        iterations=args.iterations,
        attempt_timeout=args.attempt_timeout,
        attempt_timeouts=args.attempt_timeouts,
        build_timeout=args.build_timeout,
        reflection_timeout=args.reflection_timeout,
        runner=args.runner,
        codex_bin=args.codex_bin,
        claude_bin=args.claude_bin,
        claude_model=args.claude_model,
        claude_permission_mode=args.claude_permission_mode,
        claude_max_budget_usd=args.claude_max_budget_usd,
        claude_extra_args=args.claude_extra_arg,
        run_id=args.run_id,
        build_after_solve=args.build_after_solve,
        stop_score=args.stop_score,
        score_key=args.score_key,
        higher_is_better=args.higher_is_better,
        task_state=args.task_state,
        harness_depth=args.harness_depth,
        post_attempt_command=args.post_attempt_command,
        post_attempt_score_key=args.post_attempt_score_key,
        post_attempt_timeout=args.post_attempt_timeout,
        optimization_mode=args.optimization_mode,
        restore_best=args.restore_best,
        qualify_artifacts=args.qualify_artifacts,
        artifact_repair_attempts=args.artifact_repair_attempts,
    )
    orchestrator = Orchestrator()
    results = orchestrator.run(config)
    final = results[-1] if results else None
    print(f"Run id: {orchestrator.run_id}")
    print(f"Run artifacts: {orchestrator.run_dir}")
    if final:
        print(f"Final status: {final.status}")
        print(f"Solved and verified: {final.solved}")
        print(f"Final result: {final.result_path}")
    summary = orchestrator.run_dir / "summary.json" if orchestrator.run_dir else None
    if args.optimization_mode and summary and summary.exists():
        from .artifacts import read_json

        optimization = read_json(summary).get("optimization", {})
        print(f"Optimization improved: {optimization.get('improved')}")
        print(f"Best score: {optimization.get('best_score')}")
    return 0


def compare_command(args: argparse.Namespace) -> int:
    config = CompareConfig.from_values(
        workspace_template=args.workspace_template,
        output_dir=args.output_dir,
        task=args.task,
        task_text=args.task_text,
        artifact_source=args.artifact_source,
        trials=args.trials,
        iterations=args.iterations,
        attempt_timeout=args.attempt_timeout,
        attempt_timeouts=args.attempt_timeouts,
        build_timeout=args.build_timeout,
        reflection_timeout=args.reflection_timeout,
        runner=args.runner,
        codex_bin=args.codex_bin,
        claude_bin=args.claude_bin,
        claude_model=args.claude_model,
        claude_permission_mode=args.claude_permission_mode,
        claude_max_budget_usd=args.claude_max_budget_usd,
        claude_extra_args=args.claude_extra_arg,
        stop_score=args.stop_score,
        score_key=args.score_key,
        higher_is_better=args.higher_is_better,
        task_state=args.task_state,
        build_after_solve=args.build_after_solve,
        post_command=args.post_command,
        post_score_key=args.post_score_key,
        post_timeout=args.post_timeout,
        require_active_harness=args.require_active_harness,
        require_harness_tool_use=args.require_harness_tool_use,
        overwrite=args.overwrite,
    )
    report = run_compare(config)
    print(f"Compare report: {config.output_dir / 'compare_report.json'}")
    for group, summary in report["summary"].items():
        attempt = summary["attempt_duration_seconds"]
        cumulative = summary["cumulative_attempt_duration_seconds"]
        post = summary["post_score"]
        print(
            f"{group}: n={summary['count']} "
            f"median_attempt={attempt['median']} "
            f"median_cumulative_attempt={cumulative['median']} "
            f"best_post_score={post['best']} "
            f"post_valid={post.get('valid_count', post.get('count'))} "
            f"post_invalid={post.get('invalid_count', 0)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"harnessgym: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
