#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO


SERVER_PATH = Path(__file__).resolve()
ROOT = SERVER_PATH.parents[3]
HARNESS_DIR = ROOT / ".harnessgym"
HISTORY_DIR = HARNESS_DIR / "history"
FIXTURE_DIR = HARNESS_DIR / "fixtures"
PLAN_PATH = ROOT / "kernel_plan.json"
BENCHMARK_PATH = ROOT / "benchmark.py"

sys.path.insert(0, str(ROOT))
import benchmark  # type: ignore  # noqa: E402


TOOL_NAMES = [
    "run_objective",
    "benchmark_plan",
    "validate_plan",
    "numerical_check",
    "trace_summary",
    "search_plans",
    "apply_candidate",
    "compare_history",
    "rank_next_experiments",
    "local_neighborhood_search",
    "bounded_exhaustive_search",
    "apply_best_verified",
    "candidate_diff",
    "export_candidate_fixture",
    "resume_search_history",
]


def json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def ordered_plan(plan: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(benchmark.DEFAULT_PLAN)
    normalized.update(plan)
    return {key: normalized[key] for key in benchmark.VALID}


def read_current_plan() -> dict[str, Any]:
    return ordered_plan(json.loads(PLAN_PATH.read_text(encoding="utf-8")))


def load_fixture_plan(name: str) -> dict[str, Any]:
    safe_name = name.replace("/", "_")
    path = FIXTURE_DIR / f"{safe_name}.json"
    if not path.exists():
        raise ValueError(f"unknown fixture {name!r}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "plan" in data:
        data = data["plan"]
    return ordered_plan(data)


def plan_from_args(args: dict[str, Any]) -> dict[str, Any]:
    if "plan" in args and args["plan"] is not None:
        return ordered_plan(args["plan"])
    if "fixture" in args and args["fixture"]:
        return load_fixture_plan(str(args["fixture"]))
    return read_current_plan()


def evaluate(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    result = benchmark.evaluate_plan(ordered_plan(plan), mode)
    return json_clone(result)


def trace_for(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    cases = [benchmark.case_breakdown(ordered_plan(plan), case) for case in benchmark.CASES[mode]]
    summary = benchmark.evaluate_plan(ordered_plan(plan), mode)
    return {
        "mode": mode,
        "summary": json_clone(summary),
        "cases": json_clone(cases),
    }


def validate_plan_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    errors = benchmark.validate_plan(plan)
    modes = args.get("modes", ["dev", "final"])
    mode_results = {mode: evaluate(plan, mode) for mode in modes}
    return {
        "ok": not errors,
        "errors": errors,
        "plan": plan,
        "modes": {
            mode: {
                "status": result["status"],
                "best_cycles": result["best_cycles"],
                "max_abs": result["max_abs"],
                "errors": result["errors"],
            }
            for mode, result in mode_results.items()
        },
    }


def run_objective_tool(args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode", "dev"))
    if mode not in benchmark.CASES:
        raise ValueError(f"mode must be one of {sorted(benchmark.CASES)}")
    command = [sys.executable, "benchmark.py", "--json", "--mode", mode]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    payload = json.loads(result.stdout) if result.stdout.strip().startswith("{") else None
    return {
        "command": " ".join(command),
        "mode": mode,
        "return_code": result.returncode,
        "status": payload.get("status") if payload else "failed",
        "best_cycles": payload.get("best_cycles") if payload else None,
        "score": payload.get("score") if payload else None,
        "stdout": payload,
        "stderr": result.stderr,
    }


def benchmark_plan_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    modes = args.get("modes", [args.get("mode", "dev")])
    results = {mode: evaluate(plan, mode) for mode in modes}
    return {
        "plan": plan,
        "results": {
            mode: {
                "status": result["status"],
                "best_cycles": result["best_cycles"],
                "score": result["score"],
                "max_abs": result["max_abs"],
                "errors": result["errors"],
                "cases": result["cases"],
            }
            for mode, result in results.items()
        },
    }


def numerical_check_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    modes = args.get("modes", ["dev", "final"])
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for case in benchmark.CASES[mode]:
            max_abs = benchmark.estimate_error(plan, case)
            rows.append(
                {
                    "mode": mode,
                    "case": case["name"],
                    "seq": case["seq"],
                    "dim": case["dim"],
                    "heads": case["heads"],
                    "mask_density": case["mask_density"],
                    "max_abs": max_abs,
                    "tolerance": benchmark.TOLERANCE,
                    "passed": max_abs <= benchmark.TOLERANCE,
                }
            )
    return {
        "ok": all(row["passed"] for row in rows) and not benchmark.validate_plan(plan),
        "errors": benchmark.validate_plan(plan),
        "tolerance": benchmark.TOLERANCE,
        "rows": rows,
    }


def trace_summary_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    mode = str(args.get("mode", "dev"))
    trace = trace_for(plan, mode)
    cases = []
    for row in trace["cases"]:
        components = row["components"]
        positive = {
            key: value
            for key, value in components.items()
            if key.endswith("_penalty") and value
        }
        cases.append(
            {
                "name": row["name"],
                "best_cycles": row["best_cycles"],
                "status": row["status"],
                "max_abs": row["max_abs"],
                "descriptor_count": row["descriptor_count"],
                "scratchpad_need_kb": row["scratchpad_need_kb"],
                "register_pressure": row["register_pressure"],
                "pressure_limit": row["pressure_limit"],
                "throughput": components["throughput"],
                "compute_cycles": components["compute_cycles"],
                "memory_cycles": components["memory_cycles"],
                "positive_penalties": positive,
                "bank_conflicts": row["bank_conflicts"],
                "dma_penalties": row["dma_penalties"],
                "synergies": row["synergies"],
            }
        )
    return {
        "mode": mode,
        "summary": {
            "status": trace["summary"]["status"],
            "best_cycles": trace["summary"]["best_cycles"],
            "max_abs": trace["summary"]["max_abs"],
        },
        "cases": cases,
    }


BIASED_OPTIONS: dict[str, list[Any]] = {
    "block_m": [16, 32, 48, 64, 96],
    "block_n": [16, 32, 48, 64, 96, 128],
    "block_k": [32, 64, 96, 128],
    "num_warps": [2, 4, 6, 8, 8],
    "vector_width": [4, 8, 16, 16, 16],
    "q_layout": ["row_major", "blocked_m", "swizzled_mn", "swizzled_mn", "swizzled_mn"],
    "k_layout": ["row_major", "blocked_nk", "swizzled_nk", "swizzled_nk", "transposed"],
    "v_layout": ["row_major", "blocked_kd", "swizzled_kd", "swizzled_kd", "swizzled_kd"],
    "o_layout": ["row_major", "blocked_m", "streamed", "streamed", "streamed"],
    "accum": ["scalar", "pairwise", "tree", "tree", "tree"],
    "softmax": ["exact", "online", "online_renorm", "online_renorm", "approx_poly"],
    "dma_stages": [1, 2, 3, 3, 4, 4],
    "prefetch_distance": [0, 1, 2, 2, 3, 3, 4],
    "dma_burst": [32, 64, 128, 128, 256, 256],
    "split_k": [1, 1, 2, 2, 4],
    "swizzle": ["none", "xor", "cyclic", "tensorcore", "tensorcore", "tensorcore"],
    "schedule": ["serial", "overlap_k", "persistent", "persistent", "split_pipeline", "split_pipeline"],
    "epilogue": ["store_plain", "store_vector", "fused_scale", "fused_scale_mask", "fused_scale_mask"],
    "scratchpad_kb": [32, 48, 64, 96, 128, 128],
}


FOCUSED_CORE: dict[str, list[Any]] = {
    "q_layout": ["swizzled_mn"],
    "k_layout": ["swizzled_nk"],
    "v_layout": ["swizzled_kd"],
    "o_layout": ["streamed"],
    "accum": ["tree", "pairwise"],
    "softmax": ["online_renorm", "approx_poly"],
    "swizzle": ["tensorcore"],
    "schedule": ["persistent", "split_pipeline"],
    "epilogue": ["fused_scale_mask", "fused_scale", "store_vector"],
}


NUMERIC_FOCUS: dict[str, list[Any]] = {
    "block_m": [48, 64, 32, 96, 16],
    "block_n": [64, 96, 48, 128, 32, 16],
    "block_k": [64, 96, 32, 128],
    "num_warps": [4, 6, 8, 2],
    "vector_width": [16, 8, 4],
    "dma_stages": [3, 4, 2, 1],
    "prefetch_distance": [2, 3, 1, 4, 0],
    "dma_burst": [128, 256, 64, 32],
    "split_k": [1, 2, 4],
    "scratchpad_kb": [128, 96, 64, 48, 32],
}


LOCAL_HIGH_IMPACT_FIELDS = [
    "num_warps",
    "epilogue",
    "scratchpad_kb",
    "dma_burst",
    "prefetch_distance",
    "block_m",
    "block_n",
    "block_k",
    "dma_stages",
    "split_k",
    "schedule",
    "softmax",
    "accum",
]


LOCAL_GRID_FIELDS = [
    "block_m",
    "block_n",
    "block_k",
    "dma_stages",
    "prefetch_distance",
    "dma_burst",
    "num_warps",
    "epilogue",
    "scratchpad_kb",
]


PREFERRED_VALUE_ORDER: dict[str, list[Any]] = {
    "block_m": [48, 64, 32, 96, 16],
    "block_n": [96, 64, 48, 128, 32, 16],
    "block_k": [64, 96, 32, 128],
    "num_warps": [4, 6, 8, 2],
    "vector_width": [16, 8, 4],
    "q_layout": ["swizzled_mn", "blocked_m", "row_major"],
    "k_layout": ["swizzled_nk", "blocked_nk", "transposed", "row_major"],
    "v_layout": ["swizzled_kd", "blocked_kd", "row_major"],
    "o_layout": ["streamed", "blocked_m", "row_major"],
    "accum": ["tree", "pairwise", "scalar"],
    "softmax": ["online_renorm", "approx_poly", "online", "exact"],
    "dma_stages": [3, 4, 2, 1],
    "prefetch_distance": [3, 2, 4, 1, 0],
    "dma_burst": [128, 256, 64, 32],
    "split_k": [1, 2, 4],
    "swizzle": ["tensorcore", "cyclic", "xor", "none"],
    "schedule": ["persistent", "split_pipeline", "overlap_k", "serial"],
    "epilogue": ["fused_scale_mask", "fused_scale", "store_vector", "store_plain"],
    "scratchpad_kb": [128, 96, 64, 48, 32],
}


BOUNDED_SEARCH_PROFILES: dict[str, dict[str, Any]] = {
    "iteration3_dev_core": {
        "description": "Exact high-probability dev sweep used in iteration 3: swizzled/tensorcore/streamed layout family with numeric DMA/tile variations.",
        "fixed": {
            "q_layout": "swizzled_mn",
            "k_layout": "swizzled_nk",
            "v_layout": "swizzled_kd",
            "o_layout": "streamed",
            "swizzle": "tensorcore",
        },
        "space": {
            "block_m": [16, 32, 48, 64, 96],
            "block_n": [16, 32, 48, 64, 96, 128],
            "block_k": [32, 64, 96, 128],
            "num_warps": [2, 4, 6, 8],
            "vector_width": [16],
            "accum": ["tree", "pairwise"],
            "softmax": ["online_renorm", "approx_poly"],
            "dma_stages": [1, 2, 3, 4],
            "prefetch_distance": [0, 1, 2, 3, 4],
            "dma_burst": [64, 128, 256],
            "split_k": [1, 2],
            "schedule": ["persistent", "split_pipeline"],
            "epilogue": ["fused_scale", "fused_scale_mask"],
            "scratchpad_kb": [64, 96, 128],
        },
    },
    "iteration3_layout_relaxed": {
        "description": "Bounded layout-relaxed sweep around the iteration-2/3 winning tile region, including near-optimal layout and epilogue alternatives.",
        "fixed": {},
        "space": {
            "block_m": [48, 64],
            "block_n": [64, 96],
            "block_k": [64, 96],
            "num_warps": [4, 6, 8],
            "vector_width": [16],
            "q_layout": ["swizzled_mn", "blocked_m"],
            "k_layout": ["swizzled_nk", "blocked_nk", "transposed"],
            "v_layout": ["swizzled_kd", "blocked_kd"],
            "o_layout": ["streamed", "blocked_m"],
            "accum": ["tree", "pairwise"],
            "softmax": ["online_renorm", "approx_poly"],
            "dma_stages": [3, 4],
            "prefetch_distance": [2, 3, 4],
            "dma_burst": [128, 256],
            "split_k": [1, 2],
            "swizzle": ["tensorcore", "cyclic"],
            "schedule": ["persistent", "split_pipeline"],
            "epilogue": ["fused_scale", "fused_scale_mask", "store_vector"],
            "scratchpad_kb": [64, 96, 128],
        },
    },
}


def candidate_score(plan: dict[str, Any], mode: str) -> tuple[int | None, dict[str, Any]]:
    errors = benchmark.validate_plan(plan)
    if errors:
        return None, {"status": "failed", "errors": errors}
    result = evaluate(plan, mode)
    if result["status"] != "passed":
        return None, result
    return int(result["best_cycles"]), result


def plan_key(plan: dict[str, Any]) -> tuple[Any, ...]:
    normalized = ordered_plan(plan)
    return tuple(normalized[k] for k in benchmark.VALID)


def ordered_values_for_field(field: str, seed_value: Any) -> list[Any]:
    values = list(PREFERRED_VALUE_ORDER.get(field, benchmark.VALID[field]))
    for value in benchmark.VALID[field]:
        if value not in values:
            values.append(value)
    if seed_value in values:
        values.remove(seed_value)
    return [seed_value] + values


def fields_from_args(args: dict[str, Any], key: str, default: list[str]) -> list[str]:
    fields = args.get(key, default)
    if not isinstance(fields, list):
        raise ValueError(f"{key} must be a list of plan field names")
    clean: list[str] = []
    for field in fields:
        if field not in benchmark.VALID:
            raise ValueError(f"unknown plan field {field!r}")
        if field not in clean:
            clean.append(str(field))
    return clean


def plan_from_prefixed_args(args: dict[str, Any], prefix: str, default_current: bool = False) -> dict[str, Any]:
    plan_key_name = f"{prefix}_plan"
    fixture_key_name = f"{prefix}_fixture"
    if plan_key_name in args and args[plan_key_name] is not None:
        return ordered_plan(args[plan_key_name])
    if fixture_key_name in args and args[fixture_key_name]:
        return load_fixture_plan(str(args[fixture_key_name]))
    if default_current:
        return read_current_plan()
    raise ValueError(f"provide {plan_key_name} or {fixture_key_name}")


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "best_cycles": result["best_cycles"],
        "score": result["score"],
        "max_abs": result["max_abs"],
        "errors": result["errors"],
        "cases": result["cases"],
    }


def changed_fields(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = ordered_plan(base)
    candidate = ordered_plan(candidate)
    return {
        key: {"from": base[key], "to": candidate[key]}
        for key in benchmark.VALID
        if base[key] != candidate[key]
    }


def safe_fixture_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    clean = clean.strip("._-")
    if not clean:
        raise ValueError("fixture name must contain at least one alphanumeric character")
    return clean


def maybe_add_candidate(
    top: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    name: str,
    plan: dict[str, Any],
    mode: str,
    include_final: bool,
    keep: int,
) -> None:
    plan = ordered_plan(plan)
    key = tuple(plan[k] for k in benchmark.VALID)
    if key in seen:
        return
    seen.add(key)
    score, result = candidate_score(plan, mode)
    if score is None:
        return
    row: dict[str, Any] = {
        "name": name,
        "mode": mode,
        "best_cycles": score,
        "plan": plan,
        "cases": result.get("cases", []),
    }
    if include_final:
        final = evaluate(plan, "final")
        row["final"] = {
            "status": final["status"],
            "best_cycles": final["best_cycles"],
            "max_abs": final["max_abs"],
        }
    top.append(row)
    top.sort(key=lambda item: (item["best_cycles"], item.get("final", {}).get("best_cycles", 0)))
    del top[keep:]


def bounded_profile_size(profile: dict[str, Any]) -> int:
    total = 1
    for values in profile["space"].values():
        total *= len(values)
    return total


def bounded_exhaustive_search_tool(args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode", "dev"))
    if mode not in benchmark.CASES:
        raise ValueError(f"mode must be one of {sorted(benchmark.CASES)}")
    profile_name = str(args.get("profile", "iteration3_dev_core"))
    if profile_name not in BOUNDED_SEARCH_PROFILES:
        raise ValueError(f"profile must be one of {sorted(BOUNDED_SEARCH_PROFILES)}")
    profile = BOUNDED_SEARCH_PROFILES[profile_name]
    search_space_size = bounded_profile_size(profile)
    max_evals = min(int(args.get("max_evals", search_space_size)), search_space_size)
    keep = min(int(args.get("keep", 10)), 50)
    include_final = bool(args.get("include_final", True))
    record_history = bool(args.get("record_history", True))

    top: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    maybe_add_candidate(top, seen, "current", read_current_plan(), mode, include_final, keep)
    for fixture_path in sorted(FIXTURE_DIR.glob("tensor_plan_*.json")):
        try:
            maybe_add_candidate(
                top,
                seen,
                f"fixture:{fixture_path.stem}",
                load_fixture_plan(fixture_path.stem),
                mode,
                include_final,
                keep,
            )
        except Exception:
            continue

    keys = list(profile["space"])
    evaluated = 0
    for values in itertools.product(*(profile["space"][key] for key in keys)):
        if evaluated >= max_evals:
            break
        plan = dict(profile["fixed"])
        plan.update({key: value for key, value in zip(keys, values)})
        maybe_add_candidate(top, seen, f"{profile_name}:{evaluated}", plan, mode, include_final, keep)
        evaluated += 1

    result = {
        "mode": mode,
        "profile": profile_name,
        "description": profile["description"],
        "evaluated": evaluated,
        "search_space_size": search_space_size,
        "unique_candidates": len(seen),
        "objective_metric": "best_cycles",
        "lower_is_better": True,
        "top": top,
        "notes": [
            "Search is rollback-safe and does not mutate kernel_plan.json.",
            "Profiles encode bounded exhaustive sweeps from iteration 3 so fresh sessions do not need ad hoc Python.",
        ],
    }
    if record_history:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with (HISTORY_DIR / "tensor_plan_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": int(time.time()), "bounded_exhaustive_search": result}, sort_keys=True) + "\n")
    return result


def focused_stream() -> Any:
    numeric_keys = list(NUMERIC_FOCUS)
    core_keys = list(FOCUSED_CORE)
    for numeric_values in itertools.product(*(NUMERIC_FOCUS[key] for key in numeric_keys)):
        base = {key: value for key, value in zip(numeric_keys, numeric_values)}
        for core_values in itertools.product(*(FOCUSED_CORE[key] for key in core_keys)):
            plan = dict(base)
            plan.update({key: value for key, value in zip(core_keys, core_values)})
            yield ordered_plan(plan)


def search_plans_tool(args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode", "dev"))
    if mode not in benchmark.CASES:
        raise ValueError(f"mode must be one of {sorted(benchmark.CASES)}")
    strategy = str(args.get("strategy", "quick"))
    max_evals = min(int(args.get("max_evals", args.get("samples", 100000))), 2_000_000)
    keep = min(int(args.get("keep", 8)), 50)
    include_final = bool(args.get("include_final", True))
    record_history = bool(args.get("record_history", True))
    rng = random.Random(int(args.get("seed", 20260519)))

    top: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    evaluated = 0

    maybe_add_candidate(top, seen, "current", read_current_plan(), mode, include_final, keep)
    for fixture_path in sorted(FIXTURE_DIR.glob("tensor_plan_*.json")):
        try:
            maybe_add_candidate(
                top,
                seen,
                f"fixture:{fixture_path.stem}",
                load_fixture_plan(fixture_path.stem),
                mode,
                include_final,
                keep,
            )
        except Exception:
            continue

    if strategy in {"focused", "focused_grid"}:
        for plan in focused_stream():
            maybe_add_candidate(top, seen, f"focused:{evaluated}", plan, mode, include_final, keep)
            evaluated += 1
            if evaluated >= max_evals:
                break
    else:
        keys = list(benchmark.VALID)
        while evaluated < max_evals:
            plan = {key: rng.choice(BIASED_OPTIONS[key]) for key in keys}
            maybe_add_candidate(top, seen, f"random:{evaluated}", plan, mode, include_final, keep)
            evaluated += 1

    result = {
        "mode": mode,
        "strategy": strategy,
        "evaluated": evaluated,
        "unique_candidates": len(seen),
        "objective_metric": "best_cycles",
        "lower_is_better": True,
        "top": top,
    }
    if record_history:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with (HISTORY_DIR / "tensor_plan_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": int(time.time()), "search": result}, sort_keys=True) + "\n")
    return result


def backup_current_plan() -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = HISTORY_DIR / f"kernel_plan.{stamp}.json"
    shutil.copy2(PLAN_PATH, backup)
    return backup


def write_plan_atomically(plan: dict[str, Any]) -> None:
    tmp = PLAN_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ordered_plan(plan), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, PLAN_PATH)


def apply_candidate_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    mode = str(args.get("mode", "dev"))
    dry_run = bool(args.get("dry_run", True))
    require_final_pass = bool(args.get("require_final_pass", True))
    require_improvement = bool(args.get("require_improvement", True))

    current = evaluate(read_current_plan(), mode)
    candidate = evaluate(plan, mode)
    final = evaluate(plan, "final")
    errors = benchmark.validate_plan(plan)
    ok = not errors and candidate["status"] == "passed"
    if require_final_pass:
        ok = ok and final["status"] == "passed"
    if require_improvement:
        ok = ok and int(candidate["best_cycles"]) < int(current["best_cycles"])

    applied = False
    backup_path = None
    if ok and not dry_run:
        backup_path = str(backup_current_plan().relative_to(ROOT))
        write_plan_atomically(plan)
        applied = True

    return {
        "ok": ok,
        "applied": applied,
        "dry_run": dry_run,
        "backup_path": backup_path,
        "mode": mode,
        "require_final_pass": require_final_pass,
        "require_improvement": require_improvement,
        "errors": errors,
        "current": {
            "status": current["status"],
            "best_cycles": current["best_cycles"],
            "max_abs": current["max_abs"],
        },
        "candidate": {
            "status": candidate["status"],
            "best_cycles": candidate["best_cycles"],
            "max_abs": candidate["max_abs"],
        },
        "final": {
            "status": final["status"],
            "best_cycles": final["best_cycles"],
            "max_abs": final["max_abs"],
        },
        "plan": plan,
    }


def compare_history_tool(args: dict[str, Any]) -> dict[str, Any]:
    limit = int(args.get("limit", 10))
    path = HISTORY_DIR / "tensor_plan_history.jsonl"
    searches: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                searches.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    candidates: list[dict[str, Any]] = []
    for item in searches:
        for key in ("search", "local_neighborhood_search", "bounded_exhaustive_search"):
            payload = item.get(key, {})
            for row in payload.get("top", []):
                candidates.append(
                    {
                        "time": item.get("time"),
                        "source": key,
                        "name": row.get("name"),
                        "mode": row.get("mode", payload.get("mode")),
                        "best_cycles": row.get("best_cycles"),
                        "final_best_cycles": row.get("final", {}).get("best_cycles"),
                        "final_status": row.get("final", {}).get("status"),
                        "plan": row.get("plan"),
                    }
                )
    candidates.sort(key=lambda row: (row.get("best_cycles") is None, row.get("best_cycles", 10**18)))
    return {
        "history_path": str(path.relative_to(ROOT)),
        "search_count": len(searches),
        "candidate_count": len(candidates),
        "top": candidates[:limit],
    }


def rank_next_experiments_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    mode = str(args.get("mode", "dev"))
    trace = trace_summary_tool({"plan": plan, "mode": mode})
    recommendations: list[dict[str, Any]] = []
    for case in trace["cases"]:
        if case["bank_conflicts"]:
            recommendations.append(
                {
                    "priority": "high",
                    "case": case["name"],
                    "reason": "bank conflicts are directly adding cycles",
                    "try": ["swizzled_mn/swizzled_nk/swizzled_kd layouts", "tensorcore or xor swizzle"],
                    "evidence": case["bank_conflicts"],
                }
            )
        if case["dma_penalties"]:
            recommendations.append(
                {
                    "priority": "high",
                    "case": case["name"],
                    "reason": "DMA penalties indicate staging, prefetch, or burst mismatch",
                    "try": ["dma_stages=3 with prefetch_distance=2 or 3", "dma_burst=128 before 256 on sparse masks"],
                    "evidence": case["dma_penalties"],
                }
            )
        if case["register_pressure"] > case["pressure_limit"]:
            recommendations.append(
                {
                    "priority": "medium",
                    "case": case["name"],
                    "reason": "spill pressure is above the per-shape limit",
                    "try": ["reduce num_warps before vector_width", "prefer block_m 48/64 and block_n 64/96", "avoid split_k=4 unless it unlocks final large-seq wins"],
                    "evidence": {
                        "register_pressure": case["register_pressure"],
                        "pressure_limit": case["pressure_limit"],
                    },
                }
            )
    if not recommendations:
        recommendations.append(
            {
                "priority": "medium",
                "case": "all",
                "reason": "no direct penalties dominate; use focused search around known synergistic plans",
                "try": ["search_plans strategy=focused max_evals=250000", "compare dev winner against final before applying"],
                "evidence": {"best_cycles": trace["summary"]["best_cycles"]},
            }
        )
    return {
        "mode": mode,
        "objective_metric": "best_cycles",
        "lower_is_better": True,
        "summary": trace["summary"],
        "recommendations": recommendations[: int(args.get("limit", 8))],
    }


def local_neighborhood_search_tool(args: dict[str, Any]) -> dict[str, Any]:
    mode = str(args.get("mode", "dev"))
    if mode not in benchmark.CASES:
        raise ValueError(f"mode must be one of {sorted(benchmark.CASES)}")
    seed_plan = plan_from_args(args)
    seed_label = args.get("fixture") or ("explicit_plan" if args.get("plan") is not None else "current")
    strategy = str(args.get("strategy", "checkpoint"))
    if strategy not in {"single", "pair", "grid", "checkpoint"}:
        raise ValueError("strategy must be one of single, pair, grid, checkpoint")
    max_evals = min(int(args.get("max_evals", 200000)), 2_000_000)
    keep = min(int(args.get("keep", 10)), 50)
    include_final = bool(args.get("include_final", True))
    record_history = bool(args.get("record_history", True))
    single_fields = fields_from_args(args, "single_fields", LOCAL_HIGH_IMPACT_FIELDS)
    pair_fields = fields_from_args(args, "pair_fields", LOCAL_HIGH_IMPACT_FIELDS)
    grid_fields = fields_from_args(args, "grid_fields", LOCAL_GRID_FIELDS)

    seed_result = evaluate(seed_plan, mode)
    seed_final = evaluate(seed_plan, "final") if include_final else None
    top: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = {plan_key(seed_plan)}
    evaluated = 0

    def consider(name: str, plan: dict[str, Any], mutation_count: int) -> None:
        nonlocal evaluated
        if evaluated >= max_evals:
            return
        plan = ordered_plan(plan)
        key = plan_key(plan)
        if key in seen:
            return
        seen.add(key)
        evaluated += 1
        errors = benchmark.validate_plan(plan)
        if errors:
            return
        result = evaluate(plan, mode)
        if result["status"] != "passed":
            return
        row: dict[str, Any] = {
            "name": name,
            "mode": mode,
            "best_cycles": result["best_cycles"],
            "score": result["score"],
            "mutation_count": mutation_count,
            "changed_fields": changed_fields(seed_plan, plan),
            "result": summarize_result(result),
            "plan": plan,
        }
        if include_final and (len(top) < keep or int(result["best_cycles"]) < int(top[-1]["best_cycles"])):
            final = evaluate(plan, "final")
            row["final"] = summarize_result(final)
        top.append(row)
        top.sort(key=lambda item: (int(item["best_cycles"]), item.get("final", {}).get("best_cycles", 0)))
        del top[keep:]

    if strategy in {"single", "checkpoint"}:
        for field in single_fields:
            for value in ordered_values_for_field(field, seed_plan[field])[1:]:
                plan = dict(seed_plan)
                plan[field] = value
                consider(f"single:{field}={value}", plan, 1)
                if evaluated >= max_evals:
                    break
            if evaluated >= max_evals:
                break

    if strategy in {"pair", "checkpoint"} and evaluated < max_evals:
        for left_index, left in enumerate(pair_fields):
            for right in pair_fields[left_index + 1 :]:
                for left_value in ordered_values_for_field(left, seed_plan[left])[1:]:
                    for right_value in ordered_values_for_field(right, seed_plan[right])[1:]:
                        plan = dict(seed_plan)
                        plan[left] = left_value
                        plan[right] = right_value
                        consider(f"pair:{left}={left_value},{right}={right_value}", plan, 2)
                        if evaluated >= max_evals:
                            break
                    if evaluated >= max_evals:
                        break
                if evaluated >= max_evals:
                    break
            if evaluated >= max_evals:
                break

    if strategy in {"grid", "checkpoint"} and evaluated < max_evals:
        value_lists = [ordered_values_for_field(field, seed_plan[field]) for field in grid_fields]
        for values in itertools.product(*value_lists):
            plan = dict(seed_plan)
            for field, value in zip(grid_fields, values):
                plan[field] = value
            mutation_count = sum(1 for field in grid_fields if plan[field] != seed_plan[field])
            if mutation_count == 0:
                continue
            consider("grid:" + ",".join(f"{field}={plan[field]}" for field in grid_fields if plan[field] != seed_plan[field]), plan, mutation_count)
            if evaluated >= max_evals:
                break

    history_path = None
    if record_history:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history_path = HISTORY_DIR / "tensor_plan_history.jsonl"
        payload = {
            "time": int(time.time()),
            "local_neighborhood_search": {
                "seed": seed_label,
                "mode": mode,
                "strategy": strategy,
                "evaluated": evaluated,
                "unique_candidates": len(seen),
                "top": top,
            },
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    return {
        "seed": {
            "source": seed_label,
            "mode": mode,
            "result": summarize_result(seed_result),
            "final": summarize_result(seed_final) if seed_final else None,
        },
        "mode": mode,
        "strategy": strategy,
        "evaluated": evaluated,
        "unique_candidates": len(seen),
        "objective_metric": "best_cycles",
        "lower_is_better": True,
        "top": top,
        "history_path": str(history_path.relative_to(ROOT)) if history_path else None,
        "notes": [
            "Search is rollback-safe and does not mutate kernel_plan.json.",
            "Use apply_best_verified on a returned plan before mutating the workspace.",
        ],
    }


def apply_best_verified_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    mode = str(args.get("mode", "dev"))
    if mode not in benchmark.CASES:
        raise ValueError(f"mode must be one of {sorted(benchmark.CASES)}")
    dry_run = bool(args.get("dry_run", True))
    require_improvement = bool(args.get("require_improvement", True))
    require_final_pass = bool(args.get("require_final_pass", True))

    current_plan = read_current_plan()
    current = evaluate(current_plan, mode)
    candidate = evaluate(plan, mode)
    final = evaluate(plan, "final")
    numerical = numerical_check_tool({"plan": plan, "modes": ["dev", "final"]})
    errors = benchmark.validate_plan(plan)
    ok = not errors and candidate["status"] == "passed" and numerical["ok"]
    if require_final_pass:
        ok = ok and final["status"] == "passed"
    if require_improvement:
        ok = ok and int(candidate["best_cycles"]) < int(current["best_cycles"])

    applied = False
    backup_path = None
    if ok and not dry_run:
        backup_path = str(backup_current_plan().relative_to(ROOT))
        write_plan_atomically(plan)
        applied = True

    return {
        "ok": ok,
        "applied": applied,
        "dry_run": dry_run,
        "backup_path": backup_path,
        "mode": mode,
        "require_improvement": require_improvement,
        "require_final_pass": require_final_pass,
        "errors": errors,
        "current": summarize_result(current),
        "candidate": summarize_result(candidate),
        "final": summarize_result(final),
        "numerical": numerical,
        "changed_fields": changed_fields(current_plan, plan),
        "plan": ordered_plan(plan),
    }


def candidate_diff_tool(args: dict[str, Any]) -> dict[str, Any]:
    baseline = plan_from_prefixed_args(args, "baseline", default_current=True)
    if args.get("candidate_plan") is None and not args.get("candidate_fixture") and (args.get("plan") is not None or args.get("fixture")):
        candidate = plan_from_args(args)
    else:
        candidate = plan_from_prefixed_args(args, "candidate")
    modes = args.get("modes", ["dev", "final"])
    if not isinstance(modes, list) or any(mode not in benchmark.CASES for mode in modes):
        raise ValueError(f"modes must be a list containing only {sorted(benchmark.CASES)}")
    rows: dict[str, Any] = {}
    for mode in modes:
        baseline_result = evaluate(baseline, mode)
        candidate_result = evaluate(candidate, mode)
        rows[mode] = {
            "baseline": summarize_result(baseline_result),
            "candidate": summarize_result(candidate_result),
            "delta_best_cycles": int(candidate_result["best_cycles"]) - int(baseline_result["best_cycles"]),
            "case_deltas": [
                {
                    "name": cand_case["name"],
                    "delta_best_cycles": int(cand_case["best_cycles"]) - int(base_case["best_cycles"]),
                    "baseline_best_cycles": base_case["best_cycles"],
                    "candidate_best_cycles": cand_case["best_cycles"],
                    "baseline_status": base_case["status"],
                    "candidate_status": cand_case["status"],
                }
                for base_case, cand_case in zip(baseline_result["cases"], candidate_result["cases"])
            ],
        }
    return {
        "changed_fields": changed_fields(baseline, candidate),
        "modes": rows,
        "objective_metric": "best_cycles",
        "lower_is_better": True,
    }


def export_candidate_fixture_tool(args: dict[str, Any]) -> dict[str, Any]:
    plan = plan_from_args(args)
    name = safe_fixture_name(str(args.get("name", "")))
    overwrite = bool(args.get("overwrite", False))
    source = str(args.get("source", "export_candidate_fixture"))
    notes = args.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError("notes must be a list of strings")

    errors = benchmark.validate_plan(plan)
    dev = evaluate(plan, "dev")
    final = evaluate(plan, "final")
    numerical = numerical_check_tool({"plan": plan, "modes": ["dev", "final"]})
    ok = not errors and dev["status"] == "passed" and final["status"] == "passed" and numerical["ok"]
    if not ok:
        raise ValueError("candidate must validate and pass dev/final numerical checks before fixture export")

    path = FIXTURE_DIR / f"{name}.json"
    if path.exists() and not overwrite:
        raise ValueError(f"fixture {name!r} already exists; pass overwrite=true to replace it")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "source": source,
        "objective_metric": "best_cycles",
        "mode": "dev",
        "observed_dev_best_cycles": dev["best_cycles"],
        "observed_final_best_cycles": final["best_cycles"],
        "max_abs": {"dev": dev["max_abs"], "final": final["max_abs"]},
        "notes": notes,
        "plan": ordered_plan(plan),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "path": str(path.relative_to(ROOT)),
        "dev": summarize_result(dev),
        "final": summarize_result(final),
        "numerical": numerical,
        "plan": ordered_plan(plan),
    }


def resume_search_history_tool(args: dict[str, Any]) -> dict[str, Any]:
    limit = int(args.get("limit", 8))
    compare = compare_history_tool({"limit": max(limit, 1)})
    fixture_rows: list[dict[str, Any]] = []
    for fixture_path in sorted(FIXTURE_DIR.glob("tensor_plan_*.json")):
        try:
            plan = load_fixture_plan(fixture_path.stem)
            dev = evaluate(plan, "dev")
            final = evaluate(plan, "final")
            fixture_rows.append(
                {
                    "name": fixture_path.stem,
                    "path": str(fixture_path.relative_to(ROOT)),
                    "dev_best_cycles": dev["best_cycles"],
                    "final_best_cycles": final["best_cycles"],
                    "dev_status": dev["status"],
                    "final_status": final["status"],
                    "plan": plan,
                }
            )
        except Exception as exc:
            fixture_rows.append({"name": fixture_path.stem, "path": str(fixture_path.relative_to(ROOT)), "error": str(exc)})
    fixture_rows.sort(key=lambda row: (row.get("dev_best_cycles") is None, row.get("dev_best_cycles", 10**18)))
    current = evaluate(read_current_plan(), "dev")
    return {
        "objective_metric": "best_cycles",
        "lower_is_better": True,
        "current": summarize_result(current),
        "history": compare,
        "fixtures": fixture_rows[:limit],
        "recommended_next_tools": [
            {
                "tool": "local_neighborhood_search",
                "arguments": {
                    "fixture": fixture_rows[0]["name"] if fixture_rows and "name" in fixture_rows[0] else "tensor_plan_attempt1_best",
                    "strategy": "checkpoint",
                    "max_evals": 200000,
                    "keep": 12,
                    "include_final": True,
                },
            },
            {
                "tool": "apply_best_verified",
                "arguments": {
                    "plan": "<top local_neighborhood_search plan>",
                    "dry_run": True,
                    "require_improvement": True,
                    "require_final_pass": True,
                },
            },
        ],
    }


TOOLS: dict[str, dict[str, Any]] = {
    "run_objective": {
        "description": "Run benchmark.py for the current kernel_plan.json in dev or final mode.",
        "inputSchema": {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"}},
        },
        "handler": run_objective_tool,
    },
    "benchmark_plan": {
        "description": "Evaluate a candidate plan without mutating kernel_plan.json.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
                "modes": {"type": "array", "items": {"type": "string", "enum": ["dev", "final"]}},
            },
        },
        "handler": benchmark_plan_tool,
    },
    "validate_plan": {
        "description": "Validate plan schema and summarize dev/final pass status.",
        "inputSchema": {"type": "object", "properties": {"plan": {"type": "object"}, "fixture": {"type": "string"}}},
        "handler": validate_plan_tool,
    },
    "numerical_check": {
        "description": "Check tolerance-sensitive max_abs estimates across dev and final cases.",
        "inputSchema": {"type": "object", "properties": {"plan": {"type": "object"}, "fixture": {"type": "string"}}},
        "handler": numerical_check_tool,
    },
    "trace_summary": {
        "description": "Return per-case compute/memory cycles, penalties, bank conflicts, DMA penalties, and synergies.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
            },
        },
        "handler": trace_summary_tool,
    },
    "search_plans": {
        "description": "Run rollback-safe biased or focused plan search; returns candidates and optionally records history.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
                "strategy": {"type": "string", "enum": ["quick", "focused"], "default": "quick"},
                "max_evals": {"type": "integer", "default": 100000},
                "keep": {"type": "integer", "default": 8},
                "seed": {"type": "integer", "default": 20260519},
                "include_final": {"type": "boolean", "default": True},
                "record_history": {"type": "boolean", "default": True},
            },
        },
        "handler": search_plans_tool,
    },
    "apply_candidate": {
        "description": "Rollback-safe candidate application with dev improvement and final pass gates; dry-run by default.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
                "dry_run": {"type": "boolean", "default": True},
                "require_final_pass": {"type": "boolean", "default": True},
                "require_improvement": {"type": "boolean", "default": True},
            },
        },
        "handler": apply_candidate_tool,
    },
    "compare_history": {
        "description": "Summarize recorded tensor-plan searches and top candidates.",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10}}},
        "handler": compare_history_tool,
    },
    "rank_next_experiments": {
        "description": "Rank next kernel-plan experiments from trace penalties, pressure, and DMA evidence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
                "limit": {"type": "integer", "default": 8},
            },
        },
        "handler": rank_next_experiments_tool,
    },
    "local_neighborhood_search": {
        "description": "Rollback-safe exact local search around current, fixture, or explicit plans; covers single, pair, and bounded grid mutations and records top dev candidates.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
                "strategy": {"type": "string", "enum": ["single", "pair", "grid", "checkpoint"], "default": "checkpoint"},
                "max_evals": {"type": "integer", "default": 200000},
                "keep": {"type": "integer", "default": 10},
                "include_final": {"type": "boolean", "default": True},
                "record_history": {"type": "boolean", "default": True},
                "single_fields": {"type": "array", "items": {"type": "string"}},
                "pair_fields": {"type": "array", "items": {"type": "string"}},
                "grid_fields": {"type": "array", "items": {"type": "string"}},
            },
        },
        "handler": local_neighborhood_search_tool,
    },
    "bounded_exhaustive_search": {
        "description": "Run rollback-safe bounded exhaustive sweeps captured from prior attempts, including the iteration-3 dev-core and layout-relaxed profiles.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
                "profile": {
                    "type": "string",
                    "enum": sorted(BOUNDED_SEARCH_PROFILES),
                    "default": "iteration3_dev_core",
                },
                "max_evals": {"type": "integer"},
                "keep": {"type": "integer", "default": 10},
                "include_final": {"type": "boolean", "default": True},
                "record_history": {"type": "boolean", "default": True},
            },
        },
        "handler": bounded_exhaustive_search_tool,
    },
    "apply_best_verified": {
        "description": "Apply a candidate plan only after validation, numerical checks, dev improvement, and final pass gates; dry-run by default and backs up kernel_plan.json before mutation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "mode": {"type": "string", "enum": ["dev", "final"], "default": "dev"},
                "dry_run": {"type": "boolean", "default": True},
                "require_improvement": {"type": "boolean", "default": True},
                "require_final_pass": {"type": "boolean", "default": True},
            },
        },
        "handler": apply_best_verified_tool,
    },
    "candidate_diff": {
        "description": "Compare a candidate plan against current or a baseline fixture, including changed fields and per-case dev/final cycle deltas.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "baseline_plan": {"type": "object"},
                "baseline_fixture": {"type": "string"},
                "candidate_plan": {"type": "object"},
                "candidate_fixture": {"type": "string"},
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "modes": {"type": "array", "items": {"type": "string", "enum": ["dev", "final"]}},
            },
        },
        "handler": candidate_diff_tool,
    },
    "export_candidate_fixture": {
        "description": "Save a dev/final-passing candidate plan under .harnessgym/fixtures with observed objective metadata for fresh attempts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "plan": {"type": "object"},
                "fixture": {"type": "string"},
                "source": {"type": "string"},
                "notes": {"type": "array", "items": {"type": "string"}},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["name"],
        },
        "handler": export_candidate_fixture_tool,
    },
    "resume_search_history": {
        "description": "Summarize current score, fixtures, search history, and recommended MCP calls to resume optimization without repeating prior sweeps.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 8}},
        },
        "handler": resume_search_history_tool,
    },
}


def content_response(payload: dict[str, Any], is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}],
        "isError": is_error,
    }


def read_message(stream: BinaryIO) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    line = stream.readline()
    if not line:
        return None
    while line not in {b"\r\n", b"\n", b""}:
        text = line.decode("ascii", errors="replace").strip()
        if ":" in text:
            key, value = text.split(":", 1)
            headers[key.lower()] = value.strip()
        line = stream.readline()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = stream.read(length)
    return json.loads(body.decode("utf-8"))


def write_message(stream: BinaryIO, message: dict[str, Any]) -> None:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    stream.flush()


def tool_descriptions() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": tool["description"],
            "inputSchema": tool["inputSchema"],
        }
        for name, tool in TOOLS.items()
    ]


def handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "tensor-plan-server", "version": "1.0.0"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tool_descriptions()}}
    if method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        if name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": content_response({"ok": False, "error": f"unknown tool {name!r}"}, is_error=True),
            }
        try:
            payload = TOOLS[name]["handler"](arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": content_response(payload)}
        except Exception as exc:  # Keep tool failures structured for Codex.
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": content_response({"ok": False, "error": str(exc), "tool": name}, is_error=True),
            }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def serve() -> int:
    while True:
        message = read_message(sys.stdin.buffer)
        if message is None:
            return 0
        response = handle_request(message)
        if response is not None and "id" in message:
            write_message(sys.stdout.buffer, response)


def run_self_test() -> int:
    test_path = HARNESS_DIR / "tests" / "test_tensor_plan_mcp.py"
    return subprocess.call([sys.executable, str(test_path)], cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
