#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "kernel_config.json"
KERNEL_PATH = ROOT / "moe_kernel.c"
BUILD_DIR = Path(os.environ.get("HARNESSGYM_BUILD_DIR", ROOT / ".harnessgym_build"))
TOLERANCE = 2.5e-3
TARGET_CYCLES = 850000

VALID: dict[str, list[Any]] = {
    "route_mode": ["token", "expert_scan", "bucketed"],
    "hidden_tile": [8, 16, 24, 32, 48, 64, 96, 128],
    "output_tile": [4, 8, 16, 24, 32, 48, 64],
    "input_unroll": [1, 2, 4, 8],
    "hidden_unroll": [1, 2, 4, 8],
    "prefetch": [0, 1, 2, 3, 4, 6, 8],
    "zero_mode": ["memset", "manual"],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "route_mode": "token",
    "hidden_tile": 16,
    "output_tile": 8,
    "input_unroll": 1,
    "hidden_unroll": 1,
    "prefetch": 0,
    "zero_mode": "memset",
}

ROUTE_MODE_IDS = {"token": 0, "expert_scan": 1, "bucketed": 2}
ZERO_MODE_IDS = {"memset": 0, "manual": 1}

CASES: dict[str, list[dict[str, Any]]] = {
    "dev": [
        {
            "name": "dev_096x32x064_e08_uniform",
            "tokens": 96,
            "experts": 8,
            "d_model": 32,
            "d_hidden": 64,
            "route_profile": "uniform",
            "seed": 110351,
        },
        {
            "name": "dev_160x48x096_e12_zipf",
            "tokens": 160,
            "experts": 12,
            "d_model": 48,
            "d_hidden": 96,
            "route_profile": "zipf",
            "seed": 220702,
        },
    ],
    "final": [
        {
            "name": "final_064x32x064_e08_bursty",
            "tokens": 64,
            "experts": 8,
            "d_model": 32,
            "d_hidden": 64,
            "route_profile": "bursty",
            "seed": 330053,
        },
        {
            "name": "final_160x48x096_e12_uniform",
            "tokens": 160,
            "experts": 12,
            "d_model": 48,
            "d_hidden": 96,
            "route_profile": "uniform",
            "seed": 440404,
        },
        {
            "name": "final_192x64x128_e16_zipf",
            "tokens": 192,
            "experts": 16,
            "d_model": 64,
            "d_hidden": 128,
            "route_profile": "zipf",
            "seed": 550755,
        },
        {
            "name": "final_224x80x128_e16_adversarial",
            "tokens": 224,
            "experts": 16,
            "d_model": 80,
            "d_hidden": 128,
            "route_profile": "adversarial",
            "seed": 661106,
        },
    ],
}

FORBIDDEN_SOURCE_TOKENS = (
    "PyRun_",
    "PyGILState_",
    "PyObject_",
    "dlsym",
    "RTLD_DEFAULT",
    "system(",
    "popen(",
    "fopen(",
    "open(",
    "getenv(",
    "clock_gettime",
    "mach_absolute_time",
    "__rdtsc",
)


def timer_ns(_counter: Any = time.perf_counter_ns) -> int:
    return int(_counter())


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {**DEFAULT_CONFIG, **data}


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, allowed in VALID.items():
        if config.get(key) not in allowed:
            errors.append(f"{key}={config.get(key)!r} not in {allowed!r}")
    if int(config.get("hidden_tile", 0)) > 128:
        errors.append("hidden_tile must be <= 128 for the provided stack scratch space")
    if int(config.get("output_tile", 0)) > 80:
        errors.append("output_tile must be <= max d_model")
    if config.get("route_mode") == "bucketed" and int(config.get("prefetch", 0)) > 4:
        errors.append("bucketed route mode currently supports prefetch <= 4")
    return errors


def source_integrity_errors() -> list[str]:
    text = KERNEL_PATH.read_text(encoding="utf-8", errors="replace")
    return [
        f"moe_kernel.c uses forbidden benchmark-tampering token {token!r}"
        for token in FORBIDDEN_SOURCE_TOKENS
        if token in text
    ]


def find_compiler() -> str | None:
    for candidate in ("cc", "clang", "gcc"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def compile_shared(config: dict[str, Any]) -> Path:
    compiler = find_compiler()
    if compiler is None:
        raise RuntimeError("no C compiler found; install cc, clang, or gcc")
    errors = validate_config(config) + source_integrity_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(
        KERNEL_PATH.read_bytes() + json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = BUILD_DIR / f"libcpu_moe_{source_hash}{suffix}"
    if output.exists():
        return output

    command = [
        compiler,
        "-O3",
        "-std=c11",
        "-fPIC",
        "-shared",
        "-fno-math-errno",
        f"-DHF_ROUTE_MODE={ROUTE_MODE_IDS[config['route_mode']]}",
        f"-DHF_HIDDEN_TILE={int(config['hidden_tile'])}",
        f"-DHF_OUTPUT_TILE={int(config['output_tile'])}",
        f"-DHF_INPUT_UNROLL={int(config['input_unroll'])}",
        f"-DHF_HIDDEN_UNROLL={int(config['hidden_unroll'])}",
        f"-DHF_PREFETCH={int(config['prefetch'])}",
        f"-DHF_ZERO_MODE={ZERO_MODE_IDS[config['zero_mode']]}",
        str(KERNEL_PATH),
        "-o",
        str(output),
    ]
    extra = os.environ.get("HF_MOE_EXTRA_CFLAGS")
    if extra:
        command[1:1] = extra.split()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "compile failed with command "
            + " ".join(command)
            + f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return output


def compile_assembly(config: dict[str, Any], output: Path) -> dict[str, Any]:
    compiler = find_compiler()
    if compiler is None:
        raise RuntimeError("no C compiler found; install cc, clang, or gcc")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-O3",
        "-std=c11",
        "-S",
        "-fno-math-errno",
        f"-DHF_ROUTE_MODE={ROUTE_MODE_IDS[config['route_mode']]}",
        f"-DHF_HIDDEN_TILE={int(config['hidden_tile'])}",
        f"-DHF_OUTPUT_TILE={int(config['output_tile'])}",
        f"-DHF_INPUT_UNROLL={int(config['input_unroll'])}",
        f"-DHF_HIDDEN_UNROLL={int(config['hidden_unroll'])}",
        f"-DHF_PREFETCH={int(config['prefetch'])}",
        f"-DHF_ZERO_MODE={ZERO_MODE_IDS[config['zero_mode']]}",
        str(KERNEL_PATH),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return assembly_summary(output)


def assembly_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    instruction_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith((".", "#", "//")) and not line.rstrip().endswith(":")
    ]
    return {
        "path": str(path),
        "instruction_lines": len(instruction_lines),
        "branch_mentions": sum(lowered.count(token) for token in (" j", "\tb.", "\tb\t", "cbz", "cbnz")),
        "vector_mentions": sum(lowered.count(token) for token in ("xmm", "ymm", "zmm", "\tv", ".4s", ".8h")),
        "fma_mentions": sum(lowered.count(token) for token in ("fma", "fmadd", "vfma", "vfmadd")),
        "prefetch_mentions": sum(lowered.count(token) for token in ("prfm", "prefetcht", "prefetch")),
        "load_store_mentions": sum(lowered.count(token) for token in ("ldr", "str", "mov", "load", "store")),
    }


def load_kernel(config: dict[str, Any]) -> Any:
    library = ctypes.CDLL(str(compile_shared(config)))
    fn = library.moe_forward
    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int32)
    fn.argtypes = [
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        int_ptr,
        int_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = None
    return fn


def make_case_data(case: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(int(case["seed"]))
    tokens = int(case["tokens"])
    experts = int(case["experts"])
    d_model = int(case["d_model"])
    d_hidden = int(case["d_hidden"])
    input_data = [rng.uniform(-1.0, 1.0) for _ in range(tokens * d_model)]
    w1_scale = 0.16 / math.sqrt(d_model)
    w2_scale = 0.16 / math.sqrt(d_hidden)
    w1 = [rng.uniform(-w1_scale, w1_scale) for _ in range(experts * d_hidden * d_model)]
    b1 = [rng.uniform(-0.025, 0.025) for _ in range(experts * d_hidden)]
    w2 = [rng.uniform(-w2_scale, w2_scale) for _ in range(experts * d_model * d_hidden)]
    b2 = [rng.uniform(-0.025, 0.025) for _ in range(experts * d_model)]
    top1, top2, gate1, gate2 = make_routes(rng, tokens, experts, str(case["route_profile"]))
    expected = reference_moe(input_data, w1, b1, w2, b2, top1, top2, gate1, gate2, tokens, experts, d_model, d_hidden)
    return {
        "input": input_data,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "top1": top1,
        "top2": top2,
        "gate1": gate1,
        "gate2": gate2,
        "expected": expected,
    }


def make_routes(
    rng: random.Random,
    tokens: int,
    experts: int,
    profile: str,
) -> tuple[list[int], list[int], list[float], list[float]]:
    top1: list[int] = []
    top2: list[int] = []
    gate1: list[float] = []
    gate2: list[float] = []
    zipf_weights = [1.0 / ((index + 1) ** 1.35) for index in range(experts)]

    def sample_zipf() -> int:
        return rng.choices(range(experts), weights=zipf_weights, k=1)[0]

    for token in range(tokens):
        if profile == "uniform":
            first = rng.randrange(experts)
            second = rng.randrange(experts - 1)
            if second >= first:
                second += 1
        elif profile == "zipf":
            first = sample_zipf()
            second = sample_zipf()
            if second == first:
                second = (first + 1 + rng.randrange(experts - 1)) % experts
        elif profile == "bursty":
            hot = (token // max(1, tokens // 4)) % max(1, min(experts, 4))
            first = hot if rng.random() < 0.82 else rng.randrange(experts)
            second = (hot + 1 + rng.randrange(max(1, experts - 1))) % experts
            if second == first:
                second = (second + 1) % experts
        elif profile == "adversarial":
            if token % 11 in {0, 1, 2, 3, 4, 5, 6}:
                first = 0
                second = 1 + (token % max(1, experts - 1))
            elif token % 11 in {7, 8}:
                first = min(experts - 1, 2)
                second = 0
            else:
                first = rng.randrange(experts)
                second = rng.randrange(experts - 1)
                if second >= first:
                    second += 1
        else:
            raise ValueError(f"unknown route profile {profile!r}")
        if second == first:
            second = (second + 1) % experts
        alpha = 0.54 + 0.35 * rng.random()
        top1.append(first)
        top2.append(second)
        gate1.append(alpha)
        gate2.append(1.0 - alpha)
    return top1, top2, gate1, gate2


def reference_moe(
    input_data: list[float],
    w1: list[float],
    b1: list[float],
    w2: list[float],
    b2: list[float],
    top1: list[int],
    top2: list[int],
    gate1: list[float],
    gate2: list[float],
    tokens: int,
    experts: int,
    d_model: int,
    d_hidden: int,
) -> list[float]:
    del experts
    output = [0.0] * (tokens * d_model)
    for token in range(tokens):
        x_base = token * d_model
        y_base = token * d_model
        for expert, gate in ((top1[token], gate1[token]), (top2[token], gate2[token])):
            hidden = [0.0] * d_hidden
            w1_base = expert * d_hidden * d_model
            b1_base = expert * d_hidden
            for h in range(d_hidden):
                total = b1[b1_base + h]
                row = w1_base + h * d_model
                for d in range(d_model):
                    total += input_data[x_base + d] * w1[row + d]
                hidden[h] = total if total > 0.0 else 0.0
            w2_base = expert * d_model * d_hidden
            b2_base = expert * d_model
            for d in range(d_model):
                total = b2[b2_base + d]
                row = w2_base + d * d_hidden
                for h in range(d_hidden):
                    total += hidden[h] * w2[row + h]
                output[y_base + d] += gate * total
    return output


def to_float_array(values: list[float]) -> Any:
    return (ctypes.c_float * len(values))(*values)


def to_int_array(values: list[int]) -> Any:
    return (ctypes.c_int32 * len(values))(*values)


def max_abs_error(actual: Any, expected: list[float]) -> float:
    return max(abs(float(actual[index]) - expected[index]) for index in range(len(expected)))


def evaluate_case(fn: Any, config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    data = make_case_data(case)
    tokens = int(case["tokens"])
    experts = int(case["experts"])
    d_model = int(case["d_model"])
    d_hidden = int(case["d_hidden"])
    output_size = tokens * d_model
    repeats = repeats_for_case(case)
    samples: list[int] = []
    max_abs_values: list[float] = []
    failures: list[dict[str, Any]] = []

    input_arr = to_float_array(data["input"])
    w1_arr = to_float_array(data["w1"])
    b1_arr = to_float_array(data["b1"])
    w2_arr = to_float_array(data["w2"])
    b2_arr = to_float_array(data["b2"])
    top1_arr = to_int_array(data["top1"])
    top2_arr = to_int_array(data["top2"])
    gate1_arr = to_float_array(data["gate1"])
    gate2_arr = to_float_array(data["gate2"])
    expected = data["expected"]

    for sample_index in range(2):
        dst = (ctypes.c_float * output_size)()
        fn(
            input_arr,
            w1_arr,
            b1_arr,
            w2_arr,
            b2_arr,
            top1_arr,
            top2_arr,
            gate1_arr,
            gate2_arr,
            dst,
            tokens,
            experts,
            d_model,
            d_hidden,
        )
        max_abs = max_abs_error(dst, expected)
        max_abs_values.append(max_abs)
        if max_abs > TOLERANCE:
            failures.append({"sample": sample_index, "phase": "warmup", "max_abs": max_abs})

    for sample_index in range(repeats):
        dst = (ctypes.c_float * output_size)()
        started = timer_ns()
        fn(
            input_arr,
            w1_arr,
            b1_arr,
            w2_arr,
            b2_arr,
            top1_arr,
            top2_arr,
            gate1_arr,
            gate2_arr,
            dst,
            tokens,
            experts,
            d_model,
            d_hidden,
        )
        elapsed = max(0, timer_ns() - started)
        samples.append(elapsed)
        max_abs = max_abs_error(dst, expected)
        max_abs_values.append(max_abs)
        if max_abs > TOLERANCE:
            failures.append({"sample": sample_index, "phase": "timed", "max_abs": max_abs})

    best_ns = min(samples)
    median_ns = sorted(samples)[len(samples) // 2]
    best_cycles = int(best_ns * 3.2)
    max_abs = max(max_abs_values) if max_abs_values else math.inf
    status = "passed" if not failures and max_abs <= TOLERANCE and best_ns > 0 else "failed"
    trace = case_trace(config, case, data["top1"], data["top2"])
    return {
        "name": case["name"],
        "status": status,
        "tokens": tokens,
        "experts": experts,
        "d_model": d_model,
        "d_hidden": d_hidden,
        "route_profile": case["route_profile"],
        "seed": case["seed"],
        "best_ns": best_ns,
        "median_ns": median_ns,
        "best_cycles": best_cycles,
        "score": best_cycles,
        "cycles_per_token": best_cycles / tokens,
        "max_abs": max_abs,
        "tolerance": TOLERANCE,
        "target_cycles": TARGET_CYCLES,
        "repeats": repeats,
        "validated_samples": repeats + 2,
        "failures": failures[:5],
        "route_summary": trace["route_summary"],
        "work_estimate": trace["work_estimate"],
    }


def repeats_for_case(case: dict[str, Any]) -> int:
    tokens = int(case["tokens"])
    d_model = int(case["d_model"])
    d_hidden = int(case["d_hidden"])
    ops = tokens * 4 * d_model * d_hidden
    return max(3, min(18, int(22_000_000 / max(1, ops))))


def route_summary(top1: list[int], top2: list[int], experts: int) -> dict[str, Any]:
    counts = [0] * experts
    for first, second in zip(top1, top2):
        counts[first] += 1
        counts[second] += 1
    sorted_counts = sorted(counts)
    total = sum(counts)
    mean = total / experts if experts else 0.0
    gini_num = sum(abs(a - b) for a in counts for b in counts)
    gini = gini_num / (2 * experts * total) if experts and total else 0.0
    return {
        "counts": counts,
        "empty_experts": sum(1 for count in counts if count == 0),
        "max_load": max(counts) if counts else 0,
        "min_load": min(counts) if counts else 0,
        "mean_load": mean,
        "p90_load": sorted_counts[int(0.9 * (len(sorted_counts) - 1))] if sorted_counts else 0,
        "gini": gini,
        "top_experts": sorted(
            [{"expert": expert, "routes": count} for expert, count in enumerate(counts)],
            key=lambda row: row["routes"],
            reverse=True,
        )[:5],
    }


def case_trace(config: dict[str, Any], case: dict[str, Any], top1: list[int], top2: list[int]) -> dict[str, Any]:
    tokens = int(case["tokens"])
    experts = int(case["experts"])
    d_model = int(case["d_model"])
    d_hidden = int(case["d_hidden"])
    routes = tokens * 2
    math_ops = routes * (d_model * d_hidden * 2 + d_model * d_hidden * 2)
    read_bytes = (
        tokens * d_model * 4
        + experts * d_hidden * d_model * 4
        + experts * d_model * d_hidden * 4
        + routes * (d_hidden + d_model) * 4
    )
    scratch_bytes = d_hidden * 4
    return {
        "config": config,
        "route_summary": route_summary(top1, top2, experts),
        "work_estimate": {
            "routes": routes,
            "math_ops": math_ops,
            "approx_read_bytes": read_bytes,
            "scratch_bytes_per_route": scratch_bytes,
            "route_mode": config["route_mode"],
            "hidden_tile": config["hidden_tile"],
            "output_tile": config["output_tile"],
            "input_unroll": config["input_unroll"],
            "hidden_unroll": config["hidden_unroll"],
            "prefetch": config["prefetch"],
        },
    }


def evaluate_config(config: dict[str, Any], mode: str, trace_path: Path | None = None) -> dict[str, Any]:
    if mode not in CASES:
        raise ValueError(f"unknown mode {mode!r}")
    errors = validate_config(config) + source_integrity_errors()
    if errors:
        cases = []
        status = "failed"
    else:
        fn = load_kernel(config)
        cases = [evaluate_case(fn, config, case) for case in CASES[mode]]
        status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    best_cycles = max([case["best_cycles"] for case in cases] or [10**18])
    max_abs = max([case["max_abs"] for case in cases] or [math.inf])
    result = {
        "status": status,
        "mode": mode,
        "best_cycles": best_cycles,
        "score": best_cycles,
        "max_abs": max_abs,
        "tolerance": TOLERANCE,
        "target_cycles": TARGET_CYCLES,
        "errors": errors,
        "config": config,
        "cases": cases,
        "compiler": find_compiler(),
    }
    if trace_path is not None:
        trace = {
            "mode": mode,
            "config": config,
            "cases": [
                {
                    "name": case["name"],
                    "route_summary": case.get("route_summary"),
                    "work_estimate": case.get("work_estimate"),
                    "best_cycles": case.get("best_cycles"),
                    "cycles_per_token": case.get("cycles_per_token"),
                    "status": case.get("status"),
                }
                for case in cases
            ],
        }
        trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["trace_path"] = str(trace_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit JSON result.")
    parser.add_argument("--mode", choices=sorted(CASES), default="dev")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--trace", help="Write route/work trace JSON.")
    parser.add_argument("--assembly", help="Write optimized assembly and include summary.")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    trace_path = Path(args.trace) if args.trace else None
    try:
        result = evaluate_config(config, args.mode, trace_path=trace_path)
        if args.assembly:
            result["assembly"] = compile_assembly(config, Path(args.assembly))
    except Exception as exc:
        result = {
            "status": "failed",
            "mode": args.mode,
            "best_cycles": 10**18,
            "score": 10**18,
            "max_abs": math.inf,
            "tolerance": TOLERANCE,
            "target_cycles": TARGET_CYCLES,
            "errors": [str(exc)],
            "config": config,
            "cases": [],
            "compiler": find_compiler(),
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['mode']} status={result['status']} "
            f"best_cycles={result['best_cycles']} max_abs={result['max_abs']}"
        )
        for case in result.get("cases", []):
            print(
                f"  {case['name']}: {case['status']} cycles={case['best_cycles']} "
                f"max_abs={case['max_abs']:.3g} routes={case['route_summary']['max_load']}/{sum(case['route_summary']['counts'])}"
            )
        for error in result.get("errors", []):
            print(f"  error: {error}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
