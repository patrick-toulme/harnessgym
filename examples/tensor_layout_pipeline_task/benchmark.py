#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "kernel_plan.json"
TOLERANCE = 2.5e-4
TARGET_CYCLES = 135000

VALID: dict[str, list[Any]] = {
    "block_m": [16, 32, 48, 64, 96],
    "block_n": [16, 32, 48, 64, 96, 128],
    "block_k": [32, 64, 96, 128],
    "num_warps": [2, 4, 6, 8],
    "vector_width": [4, 8, 16],
    "q_layout": ["row_major", "blocked_m", "swizzled_mn"],
    "k_layout": ["row_major", "blocked_nk", "swizzled_nk", "transposed"],
    "v_layout": ["row_major", "blocked_kd", "swizzled_kd"],
    "o_layout": ["row_major", "blocked_m", "streamed"],
    "accum": ["scalar", "pairwise", "tree"],
    "softmax": ["exact", "online", "online_renorm", "approx_poly"],
    "dma_stages": [1, 2, 3, 4],
    "prefetch_distance": [0, 1, 2, 3, 4],
    "dma_burst": [32, 64, 128, 256],
    "split_k": [1, 2, 4],
    "swizzle": ["none", "xor", "cyclic", "tensorcore"],
    "schedule": ["serial", "overlap_k", "persistent", "split_pipeline"],
    "epilogue": ["store_plain", "store_vector", "fused_scale", "fused_scale_mask"],
    "scratchpad_kb": [32, 48, 64, 96, 128],
}

DEFAULT_PLAN = {key: values[0] for key, values in VALID.items()}

CASES: dict[str, list[dict[str, Any]]] = {
    "dev": [
        {"name": "dev_256x96_causal", "seq": 256, "dim": 96, "heads": 8, "mask_density": 0.58, "seed": 730123},
        {"name": "dev_192x64_dense", "seq": 192, "dim": 64, "heads": 12, "mask_density": 1.0, "seed": 12648430},
    ],
    "final": [
        {"name": "final_128x64_dense", "seq": 128, "dim": 64, "heads": 12, "mask_density": 1.0, "seed": 659918},
        {"name": "final_256x96_causal", "seq": 256, "dim": 96, "heads": 8, "mask_density": 0.58, "seed": 730123},
        {"name": "final_320x80_sliding", "seq": 320, "dim": 80, "heads": 10, "mask_density": 0.42, "seed": 1027296},
        {"name": "final_384x128_dense", "seq": 384, "dim": 128, "heads": 6, "mask_density": 1.0, "seed": 533737},
        {"name": "final_224x96_sparse", "seq": 224, "dim": 96, "heads": 16, "mask_density": 0.36, "seed": 917339},
    ],
}


def stable_jitter(seed: int, plan: dict[str, Any], amplitude: float) -> float:
    body = json.dumps({"seed": seed, "plan": plan}, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(body).digest()
    unit = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return (unit - 0.5) * amplitude


def load_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    plan = dict(DEFAULT_PLAN)
    plan.update(data)
    return plan


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, valid_values in VALID.items():
        if key not in plan:
            errors.append(f"missing key {key}")
        elif plan[key] not in valid_values:
            errors.append(f"{key}={plan[key]!r} is not one of {valid_values!r}")
    if int(plan.get("block_k", 0)) > int(plan.get("block_n", 0)) * 2:
        errors.append("block_k cannot exceed 2x block_n without descriptor replay")
    if plan.get("schedule") == "persistent" and int(plan.get("num_warps", 0)) < 4:
        errors.append("persistent schedule requires at least 4 warps")
    if plan.get("split_k") == 4 and plan.get("accum") == "scalar":
        errors.append("split_k=4 requires pairwise or tree accumulation")
    return errors


def near(value: int, target: int, scale: float) -> float:
    return math.exp(-abs(value - target) / scale)


def estimate_error(plan: dict[str, Any], case: dict[str, Any]) -> float:
    error = 4.5e-5
    if plan["softmax"] == "online":
        error += 2.0e-5
    elif plan["softmax"] == "online_renorm":
        error += 4.0e-5
    elif plan["softmax"] == "approx_poly":
        error += 1.12e-4
    if plan["accum"] == "scalar":
        error += 2.0e-5 * int(plan["split_k"])
    elif plan["accum"] == "pairwise":
        error += 8.0e-6 * int(plan["split_k"])
    else:
        error += 4.0e-6 * int(plan["split_k"])
    if int(plan["vector_width"]) == 16 and plan["accum"] == "scalar":
        error += 3.5e-5
    if int(case["dim"]) == 128 and plan["softmax"] == "approx_poly":
        error += 5.5e-5
    if plan["epilogue"] == "fused_scale_mask" and float(case["mask_density"]) < 0.5:
        error -= 1.0e-5
    return max(error, 1.0e-6)


def descriptor_count(plan: dict[str, Any], case: dict[str, Any]) -> int:
    seq = int(case["seq"])
    block_m = int(plan["block_m"])
    block_n = int(plan["block_n"])
    block_k = int(plan["block_k"])
    split_k = int(plan["split_k"])
    tiles_m = math.ceil(seq / block_m)
    tiles_n = math.ceil(seq / block_n)
    tiles_k = math.ceil(int(case["dim"]) / block_k)
    return tiles_m * tiles_n * max(1, tiles_k) * split_k


def case_breakdown(plan: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    seq = int(case["seq"])
    dim = int(case["dim"])
    heads = int(case["heads"])
    density = float(case["mask_density"])
    block_m = int(plan["block_m"])
    block_n = int(plan["block_n"])
    block_k = int(plan["block_k"])
    warps = int(plan["num_warps"])
    vector = int(plan["vector_width"])
    stages = int(plan["dma_stages"])
    prefetch = int(plan["prefetch_distance"])
    burst = int(plan["dma_burst"])
    split_k = int(plan["split_k"])
    scratchpad = int(plan["scratchpad_kb"])

    math_ops = seq * seq * dim * heads * density
    bytes_q = seq * dim * heads * 2
    bytes_kv = 2 * seq * dim * heads * 2
    bytes_scores = seq * seq * heads * density * 2
    bytes_out = seq * dim * heads * 2
    total_bytes = bytes_q + bytes_kv + bytes_scores * 0.22 + bytes_out

    q_eff = {"row_major": 0.92, "blocked_m": 1.03, "swizzled_mn": 1.10}[plan["q_layout"]]
    k_eff = {"row_major": 0.88, "blocked_nk": 1.05, "swizzled_nk": 1.13, "transposed": 0.98}[plan["k_layout"]]
    v_eff = {"row_major": 0.94, "blocked_kd": 1.06, "swizzled_kd": 1.10}[plan["v_layout"]]
    o_eff = {"row_major": 0.96, "blocked_m": 1.03, "streamed": 1.06}[plan["o_layout"]]
    accum_eff = {"scalar": 0.91, "pairwise": 1.02, "tree": 1.09}[plan["accum"]]
    softmax_eff = {"exact": 0.86, "online": 1.02, "online_renorm": 1.09, "approx_poly": 1.14}[plan["softmax"]]
    schedule_eff = {"serial": 0.86, "overlap_k": 1.05, "persistent": 1.13, "split_pipeline": 1.10}[plan["schedule"]]
    swizzle_eff = {"none": 0.93, "xor": 1.03, "cyclic": 1.05, "tensorcore": 1.14}[plan["swizzle"]]

    tile_m_eff = 0.70 + 0.22 * near(block_m, 64 if seq >= 256 else 48, 32.0)
    tile_n_eff = 0.70 + 0.24 * near(block_n, 96 if dim >= 96 else 64, 36.0)
    tile_k_eff = 0.78 + 0.18 * near(block_k, 64 if dim <= 96 else 96, 40.0)
    vector_eff = math.sqrt(vector / 4.0)
    warp_eff = 0.86 + 0.045 * min(warps, 8)
    split_eff = 1.0 + 0.035 * (split_k - 1) if seq >= 224 else 1.0 - 0.035 * (split_k - 1)
    pipeline_eff = 1.0 + 0.055 * stages + 0.028 * min(prefetch, stages + 1)
    burst_eff = {32: 0.92, 64: 1.0, 128: 1.07, 256: 1.05}[burst]

    throughput = (
        q_eff
        * k_eff
        * v_eff
        * o_eff
        * accum_eff
        * softmax_eff
        * schedule_eff
        * swizzle_eff
        * tile_m_eff
        * tile_n_eff
        * tile_k_eff
        * vector_eff
        * warp_eff
        * split_eff
        * pipeline_eff
        * burst_eff
    )

    descriptor_reuse = 0.82 + 0.05 * stages + (0.08 if plan["schedule"] in {"persistent", "split_pipeline"} else 0.0)
    compute_cycles = math_ops / max(0.1, throughput) / 17.5 + 48000
    memory_cycles = total_bytes / (30.0 * burst_eff * max(0.5, descriptor_reuse)) + 36000

    register_pressure = (
        block_m / 15.0
        + block_n / 19.0
        + block_k / 42.0
        + warps * 1.25
        + stages * 1.55
        + split_k * 1.8
        + (3.5 if plan["accum"] == "tree" else 1.7 if plan["accum"] == "pairwise" else 0.0)
        + (3.2 if plan["swizzle"] == "tensorcore" else 1.5 if plan["swizzle"] != "none" else 0.0)
    )
    pressure_limit = 27.5 if dim <= 96 else 23.5
    spill_penalty = max(0.0, register_pressure - pressure_limit) ** 2 * (2200 if dim <= 96 else 3300)

    scratchpad_need = (
        block_m * dim * 2
        + block_n * dim * 4
        + block_m * block_n * 2
        + stages * burst * 48
    ) / 1024.0
    scratchpad_penalty = max(0.0, scratchpad_need - scratchpad) ** 2 * 1350

    bank_conflicts: list[dict[str, Any]] = []

    def conflict(name: str, cycles: float) -> None:
        if cycles:
            bank_conflicts.append({"name": name, "cycles": int(cycles)})

    if plan["q_layout"] == "row_major" and vector == 16:
        conflict("row_q_vector16_stride", 26000)
    if plan["k_layout"] == "row_major" and block_n >= 96:
        conflict("row_k_large_n_tile", 31000)
    if plan["v_layout"] == "row_major" and split_k > 1:
        conflict("row_v_splitk_replay", 18000 * split_k)
    if plan["swizzle"] == "none" and block_n % 64 != 0:
        conflict("unswizzled_odd_bank_stride", 14000)
    if plan["o_layout"] == "row_major" and plan["epilogue"] != "store_plain":
        conflict("row_output_fused_epilogue", 12000)
    bank_penalty = sum(item["cycles"] for item in bank_conflicts)

    dma_penalties: list[dict[str, Any]] = []

    def dma_penalty(name: str, cycles: float) -> None:
        if cycles:
            dma_penalties.append({"name": name, "cycles": int(cycles)})

    if prefetch > stages + 1:
        dma_penalty("prefetch_exceeds_pipeline_depth", (prefetch - stages - 1) * 21000)
    if plan["schedule"] == "serial" and stages > 1:
        dma_penalty("serial_schedule_unused_stages", (stages - 1) * 8500)
    if burst == 256 and density < 0.5:
        dma_penalty("large_burst_sparse_mask_waste", 22000)
    if descriptor_count(plan, case) > 1100 and burst == 32:
        dma_penalty("many_small_dma_descriptors", 26000)
    dma_penalty_total = sum(item["cycles"] for item in dma_penalties)

    synergies: list[dict[str, Any]] = []

    def synergy(name: str, cycles: float) -> None:
        synergies.append({"name": name, "cycles": int(cycles)})

    if plan["q_layout"] == "swizzled_mn" and plan["k_layout"] == "swizzled_nk" and plan["swizzle"] == "tensorcore":
        synergy("qk_tensorcore_swizzle_alignment", -47000)
    if plan["v_layout"] == "swizzled_kd" and plan["o_layout"] == "streamed" and plan["epilogue"] in {"fused_scale", "fused_scale_mask"}:
        synergy("streamed_v_o_epilogue", -28000)
    if plan["schedule"] == "persistent" and stages == 3 and prefetch in {2, 3}:
        synergy("persistent_three_stage_dma", -39000)
    if plan["softmax"] == "online_renorm" and plan["accum"] == "tree" and vector == 16:
        synergy("vector16_tree_online_renorm", -31000)
    if density < 0.5 and plan["epilogue"] == "fused_scale_mask" and plan["schedule"] in {"persistent", "split_pipeline"}:
        synergy("sparse_mask_epilogue_schedule", -33000)
    if dim == 128 and block_k == 96 and plan["k_layout"] == "swizzled_nk":
        synergy("d128_k96_swizzled_tile", -26000)
    if seq >= 320 and split_k == 2 and plan["schedule"] == "split_pipeline":
        synergy("large_seq_split_pipeline", -24000)
    synergy_total = sum(item["cycles"] for item in synergies)

    max_abs = estimate_error(plan, case)
    correctness_penalty = 900000 if max_abs > TOLERANCE else 0
    jitter = stable_jitter(int(case["seed"]), plan, 5200)
    raw_cycles = max(compute_cycles, memory_cycles) + spill_penalty + scratchpad_penalty
    raw_cycles += bank_penalty + dma_penalty_total + synergy_total + correctness_penalty + jitter
    best_cycles = max(1, int(raw_cycles))

    return {
        "name": case["name"],
        "seq": seq,
        "dim": dim,
        "heads": heads,
        "mask_density": density,
        "seed": case["seed"],
        "status": "passed" if max_abs <= TOLERANCE else "failed",
        "best_cycles": best_cycles,
        "median_cycles": int(best_cycles * 1.018),
        "p90_cycles": int(best_cycles * 1.045),
        "target_cycles": TARGET_CYCLES,
        "max_abs": max_abs,
        "tolerance": TOLERANCE,
        "descriptor_count": descriptor_count(plan, case),
        "scratchpad_need_kb": scratchpad_need,
        "register_pressure": register_pressure,
        "pressure_limit": pressure_limit,
        "components": {
            "math_ops": int(math_ops),
            "total_bytes": int(total_bytes),
            "throughput": throughput,
            "descriptor_reuse": descriptor_reuse,
            "compute_cycles": int(compute_cycles),
            "memory_cycles": int(memory_cycles),
            "spill_penalty": int(spill_penalty),
            "scratchpad_penalty": int(scratchpad_penalty),
            "bank_penalty": int(bank_penalty),
            "dma_penalty": int(dma_penalty_total),
            "synergy": int(synergy_total),
            "correctness_penalty": correctness_penalty,
            "jitter": int(jitter),
        },
        "bank_conflicts": bank_conflicts,
        "dma_penalties": dma_penalties,
        "synergies": synergies,
    }


def evaluate_plan(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in CASES:
        raise ValueError(f"unknown mode {mode!r}")
    errors = validate_plan(plan)
    case_rows = [case_breakdown(plan, case) for case in CASES[mode]]
    status = "passed" if not errors and all(row["status"] == "passed" for row in case_rows) else "failed"
    best_cycles = max(row["best_cycles"] for row in case_rows)
    max_abs = max(row["max_abs"] for row in case_rows)
    return {
        "mode": mode,
        "status": status,
        "errors": errors,
        "best_cycles": best_cycles,
        "score": best_cycles,
        "target_cycles": TARGET_CYCLES,
        "max_abs": max_abs,
        "tolerance": TOLERANCE,
        "plan": plan,
        "cases": [
            {
                key: row[key]
                for key in (
                    "name",
                    "seq",
                    "dim",
                    "heads",
                    "mask_density",
                    "best_cycles",
                    "median_cycles",
                    "p90_cycles",
                    "max_abs",
                    "status",
                    "target_cycles",
                    "descriptor_count",
                )
            }
            for row in case_rows
        ],
    }


def write_trace(plan: dict[str, Any], mode: str, path: Path) -> None:
    trace = {
        "mode": mode,
        "valid_values": VALID,
        "plan": plan,
        "summary": evaluate_plan(plan, mode),
        "cases": [case_breakdown(plan, case) for case in CASES[mode]],
    }
    path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def human_summary(result: dict[str, Any]) -> str:
    lines = [
        f"mode={result['mode']} status={result['status']} best_cycles={result['best_cycles']} max_abs={result['max_abs']}",
    ]
    for case in result["cases"]:
        lines.append(
            f"{case['name']}: cycles={case['best_cycles']} status={case['status']} descriptors={case['descriptor_count']}"
        )
    if result["errors"]:
        lines.extend(f"error: {error}" for error in result["errors"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(CASES), default="dev")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--trace", type=Path, help="write a detailed per-case trace JSON")
    parser.add_argument("--list-valid", action="store_true")
    args = parser.parse_args(argv)

    if args.list_valid:
        print(json.dumps({"valid": VALID, "cases": CASES, "tolerance": TOLERANCE}, indent=2, sort_keys=True))
        return 0

    try:
        plan = load_plan(args.plan)
        result = evaluate_plan(plan, args.mode)
        if args.trace:
            write_trace(plan, args.mode, args.trace)
    except Exception as exc:
        result = {
            "mode": args.mode,
            "status": "failed",
            "best_cycles": None,
            "score": None,
            "max_abs": None,
            "cases": [],
            "errors": [str(exc)],
            "plan_path": str(args.plan),
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(human_summary(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
