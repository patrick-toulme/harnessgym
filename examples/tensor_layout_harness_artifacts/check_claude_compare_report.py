#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the tensor-layout Claude compare report produced by HarnessGym."
    )
    parser.add_argument("report", help="Path to compare_report.json.")
    parser.add_argument("--min-active-mcp", type=int, default=1)
    parser.add_argument("--min-active-tools", type=int, default=1)
    parser.add_argument("--min-mcp-calls", type=int, default=0)
    parser.add_argument(
        "--require-harness-win",
        action="store_true",
        help="Require harnessed best post score to beat plain best post score.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Treat larger post scores as better when --require-harness-win is used.",
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    trials = report.get("trials")
    if not isinstance(trials, list) or not trials:
        errors.append("report has no trials")
        trials = []

    plain = [trial for trial in trials if trial.get("group") == "plain"]
    harnessed = [trial for trial in trials if trial.get("group") == "harnessed"]
    if not plain:
        errors.append("report has no plain trials")
    if not harnessed:
        errors.append("report has no harnessed trials")

    for trial in trials:
        label = f"{trial.get('group', 'unknown')} trial {trial.get('trial', '?')}"
        if trial.get("comparison_valid") is not True:
            errors.append(f"{label} comparison invalid: {trial.get('comparison_invalid_reason') or 'unknown reason'}")
        if trial.get("post_result") is None:
            errors.append(f"{label} did not run post-command verifier")
        elif trial.get("post_valid") is not True:
            errors.append(f"{label} post result invalid: {trial.get('post_invalid_reason') or 'unknown reason'}")
        if not isinstance(trial.get("post_score"), (int, float)):
            errors.append(f"{label} has no numeric post_score")

    for trial in harnessed:
        label = f"harnessed trial {trial.get('trial', '?')}"
        active_mcp = _best_count(trial, "active_mcp_count")
        active_tools = _best_count(trial, "active_tool_count")
        copied_artifacts = trial.get("copied_artifacts")
        if not isinstance(copied_artifacts, list) or not copied_artifacts:
            errors.append(f"{label} copied no harness artifacts")
        if active_mcp < args.min_active_mcp:
            errors.append(f"{label} activated {active_mcp} MCP servers; expected at least {args.min_active_mcp}")
        if active_tools < args.min_active_tools:
            errors.append(f"{label} activated {active_tools} tools; expected at least {args.min_active_tools}")
        mcp_calls = _mcp_call_count(trial)
        if mcp_calls < args.min_mcp_calls:
            errors.append(f"{label} recorded {mcp_calls} MCP tool calls; expected at least {args.min_mcp_calls}")

    plain_best = _best_post_score(plain, args.higher_is_better)
    harnessed_best = _best_post_score(harnessed, args.higher_is_better)
    if args.require_harness_win:
        if plain_best is None or harnessed_best is None:
            errors.append("--require-harness-win needs numeric plain and harnessed post scores")
        elif args.higher_is_better and harnessed_best <= plain_best:
            errors.append(f"harnessed best post score {harnessed_best} did not exceed plain {plain_best}")
        elif not args.higher_is_better and harnessed_best >= plain_best:
            errors.append(f"harnessed best post score {harnessed_best} did not beat plain {plain_best}")

    summary = {
        "status": "failed" if errors else "passed",
        "report": str(report_path),
        "plain_best_post_score": plain_best,
        "harnessed_best_post_score": harnessed_best,
        "harnessed_best_active_mcp_count": max([_best_count(trial, "active_mcp_count") for trial in harnessed] or [0]),
        "harnessed_best_active_tool_count": max([_best_count(trial, "active_tool_count") for trial in harnessed] or [0]),
        "harnessed_best_mcp_call_count": max([_mcp_call_count(trial) for trial in harnessed] or [0]),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


def _best_count(trial: dict[str, Any], key: str) -> int:
    candidates: list[int] = []
    harness_validation = trial.get("harness_validation")
    if isinstance(harness_validation, dict) and isinstance(harness_validation.get(key), (int, float)):
        candidates.append(int(harness_validation[key]))
    iterations = trial.get("iterations")
    if isinstance(iterations, list):
        for row in iterations:
            if isinstance(row, dict) and isinstance(row.get(key), (int, float)):
                candidates.append(int(row[key]))
    return max(candidates or [0])


def _mcp_call_count(trial: dict[str, Any]) -> int:
    candidates: list[int] = []
    if isinstance(trial.get("mcp_call_count"), (int, float)):
        candidates.append(int(trial["mcp_call_count"]))
    telemetry = trial.get("mcp_telemetry")
    if isinstance(telemetry, dict) and isinstance(telemetry.get("count"), (int, float)):
        candidates.append(int(telemetry["count"]))
    harness_validation = trial.get("harness_validation")
    if isinstance(harness_validation, dict) and isinstance(harness_validation.get("mcp_call_count"), (int, float)):
        candidates.append(int(harness_validation["mcp_call_count"]))
    return max(candidates or [0])


def _best_post_score(trials: list[dict[str, Any]], higher_is_better: bool) -> float | None:
    scores = [float(trial["post_score"]) for trial in trials if isinstance(trial.get("post_score"), (int, float))]
    if not scores:
        return None
    return max(scores) if higher_is_better else min(scores)


if __name__ == "__main__":
    raise SystemExit(main())
