from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("kernel_config.json")
TOLERANCE = 2.5e-4

VALID: dict[str, list[Any]] = {
    "tile_m": [16, 24, 32, 48, 64, 80, 96, 128],
    "tile_n": [16, 24, 32, 48, 64, 80, 96, 128],
    "vector_width": [4, 8, 16],
    "unroll_q": [1, 2, 3, 4, 5, 6, 8],
    "unroll_k": [1, 2, 3, 4, 5, 6, 8],
    "prefetch": [0, 1, 2, 3, 4, 5],
    "pipeline_stages": [1, 2, 3, 4, 5],
    "layout": ["row", "blocked", "swizzled", "interleaved"],
    "accum": ["scalar", "pairwise", "tree"],
    "exp": ["libm", "poly3", "poly4", "poly5", "lut"],
    "store": ["plain", "stream", "blocked"],
    "pack": [False, True],
    "split_k": [1, 2, 3, 4],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "tile_m": 16,
    "tile_n": 16,
    "vector_width": 4,
    "unroll_q": 1,
    "unroll_k": 1,
    "prefetch": 0,
    "pipeline_stages": 1,
    "layout": "row",
    "accum": "scalar",
    "exp": "libm",
    "store": "plain",
    "pack": False,
    "split_k": 1,
}

CASES = {
    "dev": [
        {"name": "dev_192x64", "n": 192, "d": 64, "seed": 12648430},
    ],
    "final": [
        {"name": "final_128x64", "n": 128, "d": 64, "seed": 659918},
        {"name": "final_192x64", "n": 192, "d": 64, "seed": 781847},
        {"name": "final_160x80", "n": 160, "d": 80, "seed": 1027296},
        {"name": "final_224x64", "n": 224, "d": 64, "seed": 533737},
    ],
}

LAYOUT_EFF = {"row": 1.0, "blocked": 1.24, "swizzled": 1.38, "interleaved": 1.31}
ACCUM_EFF = {"scalar": 1.0, "pairwise": 1.13, "tree": 1.21}
EXP_EFF = {"libm": 1.0, "poly3": 1.46, "poly4": 1.34, "poly5": 1.20, "lut": 1.29}
STORE_EFF = {"plain": 1.0, "stream": 1.08, "blocked": 1.16}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {**DEFAULT_CONFIG, **data}


def validate_config(config: dict[str, Any]) -> list[str]:
    errors = []
    for key, allowed in VALID.items():
        if config.get(key) not in allowed:
            errors.append(f"{key}={config.get(key)!r} not in {allowed!r}")
    return errors


def estimate_error(config: dict[str, Any], case: dict[str, Any]) -> float:
    exp_error = {"libm": 0.0, "poly3": 1.95e-4, "poly4": 8.5e-5, "poly5": 2.5e-5, "lut": 1.25e-4}[config["exp"]]
    accum_error = {"scalar": 5.5e-5, "pairwise": 2.5e-5, "tree": 1.2e-5}[config["accum"]]
    layout_error = 1.5e-5 if config["layout"] == "interleaved" and case["d"] == 80 else 0.0
    split_error = max(0, int(config["split_k"]) - 2) * 1.0e-5
    pack_error = -1.0e-5 if config["pack"] and config["accum"] != "scalar" else 0.0
    return max(0.0, exp_error + accum_error + layout_error + split_error + pack_error)


def _near(value: int, target: int, scale: float) -> float:
    return math.exp(-abs(value - target) / scale)


def estimate_case_cycles(config: dict[str, Any], case: dict[str, Any]) -> int:
    n = case["n"]
    d = case["d"]
    tile_m = int(config["tile_m"])
    tile_n = int(config["tile_n"])
    vector_width = int(config["vector_width"])
    unroll_q = int(config["unroll_q"])
    unroll_k = int(config["unroll_k"])
    prefetch = int(config["prefetch"])
    stages = int(config["pipeline_stages"])
    split_k = int(config["split_k"])

    math_work = n * n * d
    vector_eff = math.sqrt(vector_width / 4.0)
    unroll_eff = 1.0 + 0.075 * min(unroll_q, 5) + 0.065 * min(unroll_k, 5)
    pipe_eff = 1.0 + 0.055 * stages
    prefetch_eff = 1.0 + 0.035 * prefetch
    pack_eff = 1.11 if config["pack"] else 1.0
    split_eff = 1.0 + 0.035 * (split_k - 1)
    layout_eff = LAYOUT_EFF[config["layout"]]
    accum_eff = ACCUM_EFF[config["accum"]]
    exp_eff = EXP_EFF[config["exp"]]
    store_eff = STORE_EFF[config["store"]]

    ideal_m = 64 if d == 64 else 48
    ideal_n = 96 if n >= 192 else 64
    tile_eff = 0.72 + 0.22 * _near(tile_m, ideal_m, 28.0) + 0.24 * _near(tile_n, ideal_n, 34.0)

    throughput = vector_eff * unroll_eff * pipe_eff * prefetch_eff * pack_eff
    throughput *= split_eff * layout_eff * accum_eff * exp_eff * store_eff * tile_eff

    register_pressure = (
        tile_m / 18.0
        + tile_n / 26.0
        + unroll_q * 1.7
        + unroll_k * 1.35
        + stages * 1.2
        + split_k * 1.1
        + (4.0 if config["accum"] == "tree" else 1.5 if config["accum"] == "pairwise" else 0.0)
        + (3.0 if config["layout"] == "swizzled" else 1.8 if config["layout"] == "interleaved" else 0.0)
    )
    pressure_limit = 22.0 if d == 64 else 18.0
    spill_penalty = max(0.0, register_pressure - pressure_limit) ** 2 * (1550 if d == 64 else 2350)

    occupancy_penalty = 0.0
    if tile_m * tile_n > (8192 if d == 64 else 6144):
        occupancy_penalty += (tile_m * tile_n - (8192 if d == 64 else 6144)) * 3.8
    if prefetch > stages + 1:
        occupancy_penalty += (prefetch - stages - 1) * 9500
    if config["layout"] == "row" and tile_n > 64:
        occupancy_penalty += 24000
    if config["exp"] == "poly3" and config["accum"] == "scalar":
        occupancy_penalty += 18000
    if config["store"] == "stream" and n < 160:
        occupancy_penalty += 14000
    if split_k > 1 and n < 160:
        occupancy_penalty += 12000 * (split_k - 1)

    synergy = 0.0
    if config["layout"] == "swizzled" and config["pack"] and vector_width == 16:
        synergy -= 32000
    if config["accum"] == "tree" and config["exp"] in {"poly4", "poly5"}:
        synergy -= 22000
    if config["store"] == "blocked" and tile_n in {64, 96}:
        synergy -= 16000
    if d == 80 and config["layout"] == "interleaved" and tile_m == 48:
        synergy -= 28000
    if n >= 192 and split_k == 2 and prefetch in {2, 3}:
        synergy -= 18000
    if tile_m == 64 and tile_n == 96 and unroll_q in {4, 5} and unroll_k in {3, 4}:
        synergy -= 35000

    stable = hashlib.sha256(
        json.dumps({"seed": case["seed"], "config": config}, sort_keys=True).encode("utf-8")
    ).digest()
    deterministic_jitter = ((int.from_bytes(stable[:2], "big") / 65535.0) - 0.5) * 3500
    raw_cycles = math_work / max(0.2, throughput) / 3.65 + 72000
    raw_cycles += spill_penalty + occupancy_penalty + synergy + deterministic_jitter
    if estimate_error(config, case) > TOLERANCE:
        raw_cycles += 900000
    return max(1, int(raw_cycles))


def evaluate_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    errors = validate_config(config)
    cases = []
    for case in CASES[mode]:
        max_abs = estimate_error(config, case)
        cycles = estimate_case_cycles(config, case)
        status = "passed" if max_abs <= TOLERANCE and not errors else "failed"
        cases.append(
            {
                **case,
                "best_cycles": cycles,
                "median_cycles": int(cycles * 1.018),
                "p90_cycles": int(cycles * 1.045),
                "max_abs": max_abs,
                "target_cycles": 120000,
                "status": status,
                "errors": errors,
            }
        )
    best_cycles = max(case["best_cycles"] for case in cases)
    max_abs = max(case["max_abs"] for case in cases)
    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    return {
        "status": status,
        "mode": mode,
        "best_cycles": best_cycles,
        "score": best_cycles,
        "median_cycles": max(case["median_cycles"] for case in cases),
        "p90_cycles": max(case["p90_cycles"] for case in cases),
        "max_abs": max_abs,
        "target_cycles": 120000,
        "cases": cases,
        "config": config,
    }


def iter_local_neighbors(config: dict[str, Any]) -> list[dict[str, Any]]:
    neighbors = []
    for key, allowed in VALID.items():
        current = config[key]
        index = allowed.index(current)
        for new_index in {max(0, index - 1), min(len(allowed) - 1, index + 1)}:
            if new_index != index:
                candidate = dict(config)
                candidate[key] = allowed[new_index]
                neighbors.append(candidate)
    return neighbors


def config_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(config[key] for key in VALID)


def beam_search(mode: str, width: int = 48, rounds: int = 18) -> list[dict[str, Any]]:
    seed_configs = [
        dict(DEFAULT_CONFIG),
        {
            "tile_m": 64,
            "tile_n": 96,
            "vector_width": 16,
            "unroll_q": 4,
            "unroll_k": 4,
            "prefetch": 3,
            "pipeline_stages": 3,
            "layout": "swizzled",
            "accum": "tree",
            "exp": "poly4",
            "store": "blocked",
            "pack": True,
            "split_k": 2,
        },
        {
            "tile_m": 48,
            "tile_n": 96,
            "vector_width": 16,
            "unroll_q": 5,
            "unroll_k": 3,
            "prefetch": 3,
            "pipeline_stages": 4,
            "layout": "interleaved",
            "accum": "tree",
            "exp": "poly4",
            "store": "blocked",
            "pack": True,
            "split_k": 2,
        },
    ]
    beam = seed_configs
    seen = {config_key(config) for config in beam}
    scored: list[tuple[int, dict[str, Any]]] = []
    for _ in range(rounds):
        candidates = list(beam)
        for config in beam:
            for neighbor in iter_local_neighbors(config):
                key = config_key(neighbor)
                if key not in seen:
                    seen.add(key)
                    candidates.append(neighbor)
        scored = sorted(
            ((evaluate_config(config, mode)["score"], config) for config in candidates),
            key=lambda item: item[0],
        )
        beam = [config for _, config in scored[:width]]
    return [
        {"score": score, "config": config, "result": evaluate_config(config, mode)}
        for score, config in scored[:width]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=sorted(CASES), default="dev")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--beam-search", action="store_true")
    parser.add_argument("--write-best", action="store_true")
    args = parser.parse_args()

    if args.beam_search:
        ranked = beam_search(args.mode)
        payload = {"status": "passed", "mode": args.mode, "ranked": ranked[:10], "best": ranked[0]}
        if args.write_best:
            CONFIG_PATH.write_text(json.dumps(ranked[0]["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        payload = evaluate_config(load_config(Path(args.config)), args.mode)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{payload['status']} score={payload.get('score', payload.get('best', {}).get('score'))}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
