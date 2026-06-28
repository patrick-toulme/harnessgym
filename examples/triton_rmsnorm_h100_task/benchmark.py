#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from kernel import fused_rmsnorm_silu_gate, load_config, torch_reference, validate_config


ROOT = Path(__file__).resolve().parent
TOLERANCE = 7.5e-3
TARGET_BEST_US = 90.0


CASES: dict[str, list[dict[str, Any]]] = {
    "dev": [
        {"name": "dev_r1024_d1024", "rows": 1024, "dim": 1024, "seed": 1701},
        {"name": "dev_r768_d2048", "rows": 768, "dim": 2048, "seed": 1702},
        {"name": "dev_r384_d4096", "rows": 384, "dim": 4096, "seed": 1703},
    ],
    "final": [
        {"name": "final_r1536_d1024", "rows": 1536, "dim": 1024, "seed": 2701},
        {"name": "final_r896_d2048", "rows": 896, "dim": 2048, "seed": 2702},
        {"name": "final_r448_d4096", "rows": 448, "dim": 4096, "seed": 2703},
        {"name": "final_r192_d8192", "rows": 192, "dim": 8192, "seed": 2704},
    ],
}


def make_inputs(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(case["seed"]))
    shape = (int(case["rows"]), int(case["dim"]))
    x = torch.randn(shape, device=device, dtype=torch.float16, generator=generator).contiguous()
    gate = torch.randn(shape, device=device, dtype=torch.float16, generator=generator).contiguous()
    weight = (0.4 + torch.rand((shape[1],), device=device, dtype=torch.float16, generator=generator)).contiguous()
    return x, gate, weight


def time_call(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return times_ms


def run_case(case: dict[str, Any], config: dict[str, Any], warmup: int, repeats: int) -> dict[str, Any]:
    x, gate, weight = make_inputs(case)
    errors = validate_config(config, int(case["dim"]))
    if errors:
        return {
            "name": case["name"],
            "status": "failed",
            "errors": errors,
            "best_us": float("inf"),
            "max_abs": float("inf"),
        }

    # Compile both paths and verify before timing.
    out = fused_rmsnorm_silu_gate(x, gate, weight, config=config)
    ref = torch_reference(x, gate, weight)
    torch.cuda.synchronize()
    max_abs = float(torch.max(torch.abs(out.float() - ref.float())).item())
    passed = max_abs <= TOLERANCE

    times_ms = time_call(lambda: fused_rmsnorm_silu_gate(x, gate, weight, config=config), warmup, repeats)
    best_ms = min(times_ms)
    median_ms = statistics.median(times_ms)
    return {
        "name": case["name"],
        "status": "passed" if passed else "failed",
        "rows": int(case["rows"]),
        "dim": int(case["dim"]),
        "numel": int(case["rows"]) * int(case["dim"]),
        "max_abs": max_abs,
        "tolerance": TOLERANCE,
        "best_us": best_ms * 1000.0,
        "median_us": median_ms * 1000.0,
        "gbps_effective": effective_gbps(case, best_ms),
    }


def effective_gbps(case: dict[str, Any], best_ms: float) -> float:
    numel = int(case["rows"]) * int(case["dim"])
    # Approximate bytes touched: x load, gate load, output store, and weight load per row.
    bytes_touched = numel * 2 * 3 + int(case["rows"]) * int(case["dim"]) * 2
    seconds = best_ms / 1000.0
    return float(bytes_touched / seconds / 1.0e9) if seconds > 0 else 0.0


def run_suite(mode: str, warmup: int, repeats: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "status": "failed",
            "error": "CUDA is not available",
            "mode": mode,
            "best_us": float("inf"),
            "score": float("inf"),
        }

    torch.backends.cuda.matmul.allow_tf32 = True
    config = load_config()
    started = time.perf_counter()
    case_results = [run_case(case, config, warmup, repeats) for case in CASES[mode]]
    total_best_us = sum(float(result["best_us"]) for result in case_results)
    total_median_us = sum(float(result["median_us"]) for result in case_results if "median_us" in result)
    passed = all(result.get("status") == "passed" for result in case_results)
    return {
        "status": "passed" if passed else "failed",
        "mode": mode,
        "objective": "minimize total best_us across held-out fused RMSNorm+SiLU gate cases",
        "best_us": total_best_us,
        "score": total_best_us,
        "median_us": total_median_us,
        "target_best_us": TARGET_BEST_US,
        "config": config,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cases": case_results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the H100 Triton RMSNorm gate task.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--mode", choices=sorted(CASES), default="dev", help="Benchmark mode.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per case.")
    parser.add_argument("--repeats", type=int, default=50, help="Timing repeats per case.")
    parser.add_argument("--trace", help="Optional path to write full JSON trace.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    result = run_suite(args.mode, args.warmup, args.repeats)
    if args.trace:
        Path(args.trace).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    text = json.dumps(result, indent=None if args.json else 2, sort_keys=True)
    print(text)
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
