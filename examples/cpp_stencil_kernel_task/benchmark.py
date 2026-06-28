#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
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
BUILD_DIR = Path(os.environ.get("HARNESSGYM_BUILD_DIR", ROOT / ".harnessgym_build"))
KERNEL = ROOT / "kernel.cpp"

CASES = {
    "dev": [
        {"name": "dev_128x128_a073", "rows": 128, "cols": 128, "alpha": 0.73, "seed": 110351},
        {"name": "dev_192x160_a041", "rows": 192, "cols": 160, "alpha": 0.41, "seed": 220702},
        {"name": "dev_guard_128x128_a019", "rows": 128, "cols": 128, "alpha": 0.19, "seed": 991003},
    ],
    "final": [
        {"name": "final_96x224_a061", "rows": 96, "cols": 224, "alpha": 0.61, "seed": 330053},
        {"name": "final_256x192_a028", "rows": 256, "cols": 192, "alpha": 0.28, "seed": 440404},
        {"name": "final_320x256_a087", "rows": 320, "cols": 256, "alpha": 0.87, "seed": 550755},
        {"name": "final_guard_96x224_a033", "rows": 96, "cols": 224, "alpha": 0.33, "seed": 661106},
        {"name": "final_guard_320x256_a052", "rows": 320, "cols": 256, "alpha": 0.52, "seed": 771457},
    ],
}

FORBIDDEN_SOURCE_TOKENS = (
    "PyRun_",
    "PyGILState_",
    "PyObject_",
    "dlsym",
    "RTLD_DEFAULT",
    "perf_counter",
    "monotonic",
)


def timer_ns(_counter: Any = time.perf_counter_ns) -> int:
    return int(_counter())


def find_compiler() -> str | None:
    for candidate in ("c++", "clang++", "g++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def compile_shared() -> Path:
    compiler = find_compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler found; install c++, clang++, or g++")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    output = BUILD_DIR / f"libstencil{suffix}"
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        str(KERNEL),
        "-o",
        str(output),
    ]
    if not output.exists() or KERNEL.stat().st_mtime > output.stat().st_mtime:
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return output


def write_assembly(path: Path) -> dict[str, Any]:
    compiler = find_compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler found; install c++, clang++, or g++")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O3", "-std=c++17", "-S", str(KERNEL), "-o", str(path)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return assembly_summary(path)


def assembly_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    instruction_lines = [
        line for line in lines if not line.startswith((".", "#", "//")) and not line.endswith(":")
    ]
    lowered = "\n".join(instruction_lines).lower()
    return {
        "path": str(path),
        "instruction_lines": len(instruction_lines),
        "branch_mentions": sum(lowered.count(op) for op in (" j", "\tb", "cbz", "cbnz")),
        "vector_register_mentions": sum(lowered.count(token) for token in ("xmm", "ymm", "zmm", "\tv", " q")),
        "fma_mentions": sum(lowered.count(token) for token in ("fma", "vfma", "vfmadd", "fmadd")),
        "load_store_mentions": sum(lowered.count(token) for token in ("load", "store", "ldr", "str", "mov")),
    }


def load_kernel() -> Any:
    errors = source_integrity_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    library = ctypes.CDLL(str(compile_shared()))
    fn = library.stencil_step
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    fn.restype = None
    return fn


def source_integrity_errors() -> list[str]:
    text = KERNEL.read_text(encoding="utf-8", errors="replace")
    errors = [
        f"kernel.cpp uses forbidden benchmark-tampering token {token!r}"
        for token in FORBIDDEN_SOURCE_TOKENS
        if token in text
    ]
    return errors


def make_input(size: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(size)]


def reference(src: list[float], rows: int, cols: int, alpha: float) -> list[float]:
    out = [0.0] * (rows * cols)

    def load(r: int, c: int) -> float:
        r = min(rows - 1, max(0, r))
        c = min(cols - 1, max(0, c))
        return src[r * cols + c]

    for r in range(rows):
        for c in range(cols):
            center = load(r, c)
            filtered = 0.50 * center + 0.125 * (
                load(r - 1, c) + load(r + 1, c) + load(r, c - 1) + load(r, c + 1)
            )
            out[r * cols + c] = alpha * filtered + (1.0 - alpha) * center
    return out


def evaluate_case(fn: Any, case: dict[str, Any]) -> dict[str, Any]:
    rows = int(case["rows"])
    cols = int(case["cols"])
    alpha = float(case["alpha"])
    size = rows * cols
    repeats = max(5, min(40, int(350000 / max(1, size))))
    samples: list[int] = []
    max_abs_values: list[float] = []
    failures: list[dict[str, Any]] = []

    prepared = [
        prepare_buffers(size=size, seed=int(case["seed"]) + 7919 * sample_index, rows=rows, cols=cols, alpha=alpha)
        for sample_index in range(repeats + 3)
    ]

    for sample_index in range(3):
        src, dst, expected = prepared[sample_index]
        fn(src, dst, rows, cols, alpha)
        max_abs = max_abs_error(dst, expected)
        max_abs_values.append(max_abs)
        if max_abs > 2.5e-5:
            failures.append({"sample": sample_index, "phase": "warmup", "max_abs": max_abs})

    for sample_index in range(repeats):
        src, dst, expected = prepared[sample_index + 3]
        started = timer_ns()
        fn(src, dst, rows, cols, alpha)
        elapsed = max(0, timer_ns() - started)
        samples.append(elapsed)
        max_abs = max_abs_error(dst, expected)
        max_abs_values.append(max_abs)
        if max_abs > 2.5e-5:
            failures.append({"sample": sample_index, "phase": "timed", "max_abs": max_abs})
    best_ns = min(samples)
    median_ns = sorted(samples)[len(samples) // 2]
    estimated_cycles = int(best_ns * 3.2)
    cycles_per_cell = estimated_cycles / size
    max_abs = max(max_abs_values) if max_abs_values else math.inf
    status = "passed" if not failures and max_abs <= 2.5e-5 and best_ns > 0 else "failed"
    return {
        "name": case["name"],
        "rows": rows,
        "cols": cols,
        "alpha": alpha,
        "seed": int(case["seed"]),
        "status": status,
        "max_abs": max_abs,
        "best_ns": best_ns,
        "median_ns": median_ns,
        "best_cycles": estimated_cycles,
        "cycles_per_cell": cycles_per_cell,
        "repeats": repeats,
        "validated_samples": repeats + 3,
        "failures": failures[:5],
    }


def prepare_buffers(
    *,
    size: int,
    seed: int,
    rows: int,
    cols: int,
    alpha: float,
) -> tuple[Any, Any, list[float]]:
    src_values = make_input(size, seed)
    expected = reference(src_values, rows, cols, alpha)
    src = (ctypes.c_float * size)(*src_values)
    dst = (ctypes.c_float * size)(*([12345.0] * size))
    return src, dst, expected


def max_abs_error(dst: Any, expected: list[float]) -> float:
    return max(abs(float(dst[i]) - expected[i]) for i in range(len(expected)))


def evaluate(mode: str, assembly_path: Path | None = None) -> dict[str, Any]:
    if mode not in CASES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(CASES))}")
    fn = load_kernel()
    case_rows = [evaluate_case(fn, case) for case in CASES[mode]]
    status = "passed" if all(row["status"] == "passed" for row in case_rows) else "failed"
    best_cycles = max(row["best_cycles"] for row in case_rows)
    result: dict[str, Any] = {
        "mode": mode,
        "status": status,
        "cases": case_rows,
        "best_cycles": best_cycles,
        "score": best_cycles,
        "max_abs": max(row["max_abs"] for row in case_rows),
        "compiler": find_compiler(),
        "objective": "minimize max per-case estimated cycles while every varied timed sample keeps max_abs <= 2.5e-5",
        "source_integrity": "passed",
    }
    if assembly_path is not None:
        result["assembly"] = write_assembly(assembly_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(CASES), default="dev")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--assembly", type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.mode, args.assembly)
    except Exception as exc:
        result = {
            "mode": args.mode,
            "status": "failed",
            "errors": [str(exc)],
            "best_cycles": math.inf,
            "score": math.inf,
            "source_integrity": "failed",
        }
    if args.trace:
        args.trace.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
