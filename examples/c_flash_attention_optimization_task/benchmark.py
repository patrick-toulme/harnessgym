from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BenchCase:
    name: str
    n: int
    d: int
    runs: int
    seed: int


DEV_CASE = BenchCase("dev_192x64", 192, 64, 9, 0xC0FFEE)
FINAL_CASES = (
    BenchCase("final_128x64", 128, 64, 9, 0xA11CE),
    BenchCase("final_192x64", 192, 64, 9, 0xBEE17),
    BenchCase("final_160x80", 160, 80, 7, 0xFACE0),
)


def compile_benchmark(case: BenchCase) -> Path:
    binary = ROOT / f"flash_bench_{case.name}"
    defines = [
        f"-DN={case.n}",
        f"-DD={case.d}",
        f"-DRUNS={case.runs}",
        f"-DCASE_SEED={case.seed}u",
    ]
    candidates = [
        ["cc", "-O3", "-march=native", "-std=c11", *defines, "benchmark.c", "kernel.c", "-lm", "-o", str(binary)],
        ["cc", "-O3", "-std=c11", *defines, "benchmark.c", "kernel.c", "-lm", "-o", str(binary)],
    ]
    errors = []
    for command in candidates:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            return binary
        errors.append({"command": command, "stderr": result.stderr})
    raise RuntimeError(json.dumps(errors, indent=2))


def run_case(case: BenchCase) -> dict:
    binary = compile_benchmark(case)
    result = subprocess.run([str(binary)], cwd=ROOT, capture_output=True, text=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"benchmark did not emit JSON: stdout={result.stdout!r} stderr={result.stderr!r}") from exc
    payload["name"] = case.name
    payload["return_code"] = result.returncode
    payload["seed"] = case.seed
    if result.returncode != 0:
        raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def run_benchmark(mode: str) -> dict:
    cases = (DEV_CASE,) if mode == "dev" else FINAL_CASES
    results = [run_case(case) for case in cases]
    best_cycles = max(result["best_cycles"] for result in results)
    median_cycles = max(result["median_cycles"] for result in results)
    p90_cycles = max(result["p90_cycles"] for result in results)
    max_abs = max(result["max_abs"] for result in results)
    return {
        "status": "passed",
        "mode": mode,
        "cases": results,
        "best_cycles": best_cycles,
        "median_cycles": median_cycles,
        "p90_cycles": p90_cycles,
        "max_abs": max_abs,
        "score": best_cycles,
        "target_cycles": 350000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=("dev", "final"), default="dev")
    args = parser.parse_args()
    payload = run_benchmark(args.mode)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "verifier passed; "
            f"mode={payload['mode']}; "
            f"best_cycles={payload['best_cycles']:.0f}; "
            f"median_cycles={payload['median_cycles']:.0f}; "
            f"p90_cycles={payload['p90_cycles']:.0f}; "
            f"target_cycles={payload['target_cycles']:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
