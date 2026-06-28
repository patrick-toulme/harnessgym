#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


SERVER_NAME = "h100-triton-rmsnorm-harness"
SERVER_VERSION = "0.1.0"
HISTORY_REL = Path(".harnessgym/history/h100_rmsnorm_runs.jsonl")
ALLOWED_CONFIG_KEYS = {"num_warps", "num_stages", "block_size", "rows_per_program"}
VALID_MODES = {"dev", "final"}
VALID_WARPS = {1, 2, 4, 8, 16, 32}
VALID_STAGES = {1, 2, 3, 4, 5}
VALID_LAUNCH_DIMS = {1024, 2048, 4096, 8192}
VALID_ROWS_PER_PROGRAM = {0, 1, 2, 4, 8}
RESTORE_ALLOWLIST = {"kernel.py", "kernel_config.json"}

LAUNCH_PRESETS = {
    1024: [(4, 4, 3), (4, 2, 3), (4, 8, 3), (4, 4, 2), (2, 4, 3), (8, 4, 3)],
    2048: [(2, 4, 3), (2, 2, 3), (2, 8, 3), (2, 4, 2), (1, 4, 3), (4, 4, 3)],
    4096: [(2, 8, 3), (2, 4, 3), (2, 16, 3), (2, 8, 2), (1, 8, 3), (4, 8, 3)],
    8192: [(0, 32, 1), (0, 16, 1), (0, 32, 2), (0, 16, 2), (1, 32, 1), (2, 16, 1)],
}

LOG2_E = 1.4426950408889634
SILU_VARIANTS = {
    "exp": "gate / (1.0 + tl.exp(-gate))",
    "sigmoid": "gate * tl.sigmoid(gate)",
    "exp2": f"gate / (1.0 + tl.exp2(-{LOG2_E} * gate))",
}
SILU_APPROX_VARIANTS = {
    "rational_m2n2": {
        "num": [0.24999548, 0.00657509112, 0.0000119853133],
        "den": [1.0, 0.109603602, 0.000857215925],
        "range": 8.0,
        "notes": "Lowest arithmetic count; failed one observed final 4096 case by one fp16 step in iteration 2.",
    },
    "rational_m3n2": {
        "num": [0.249999786, 0.00703274510, 0.0000183763207, -0.0000000152714213],
        "den": [1.0, 0.11146245, 0.00102942],
        "range": 8.0,
        "notes": "Numerically safe in the iteration-2 GPU probe but regressed final latency; benchmark before keeping.",
    },
    "rational_m3n3": {
        "num": [0.249999997, 0.00789827351, 0.0000406248742, 0.0000000226123130],
        "den": [1.0, 0.114926398, 0.00140637978, 0.00000281985750],
        "range": 8.0,
        "notes": "More accurate rational approximation with extra division polynomial work.",
    },
}
APPROX_HELPER_START = "# HARNESSGYM_SILU_APPROX_START"
APPROX_HELPER_END = "# HARNESSGYM_SILU_APPROX_END"


class ToolError(Exception):
    pass


def compact_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(compact_json(data) + "\n")


def parse_cases(workspace: Path) -> dict[str, Any]:
    source_path = workspace / "benchmark.py"
    if not source_path.exists():
        return {}
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CASES":
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CASES":
            return ast.literal_eval(node.value)
    return {}


def latest_result_json(workspace: Path) -> Path | None:
    root = workspace / ".harnessgym/runs"
    if not root.exists():
        return None
    candidates = list(root.glob("*/iterations/*/result.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def best_manifest_json(workspace: Path) -> dict[str, Any] | None:
    manifests = sorted((workspace / ".harnessgym/runs").glob("*/checkpoints/best_manifest.json"))
    if not manifests:
        return None
    return read_json(manifests[-1])


def source_supports_launch_overrides(workspace: Path) -> bool:
    path = workspace / "kernel.py"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "rows_per_program_" in text and "num_warps_" in text and "num_stages_" in text


def validate_launch_override(config: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, raw_value in config.items():
        if "_" not in key:
            raise ToolError(f"launch override key must include a dim suffix: {key}")
        prefix, _, dim_text = key.rpartition("_")
        try:
            dim = int(dim_text)
            value = int(raw_value)
        except ValueError as exc:
            raise ToolError(f"launch override values and dim suffixes must be integers: {key}") from exc
        if dim not in VALID_LAUNCH_DIMS:
            raise ToolError(f"{key} dim must be one of {sorted(VALID_LAUNCH_DIMS)}")
        if prefix == "rows_per_program":
            if value not in VALID_ROWS_PER_PROGRAM:
                raise ToolError(f"{key} must be one of {sorted(VALID_ROWS_PER_PROGRAM)}")
        elif prefix == "num_warps":
            if value not in VALID_WARPS:
                raise ToolError(f"{key} must be one of {sorted(VALID_WARPS)}")
        elif prefix == "num_stages":
            if value not in VALID_STAGES:
                raise ToolError(f"{key} must be one of {sorted(VALID_STAGES)}")
        else:
            raise ToolError(f"unsupported launch override key: {key}")
        out[key] = value
    return out


def generated_launch_overrides(dims: list[int] | None = None) -> list[dict[str, int]]:
    selected_dims = dims or sorted(VALID_LAUNCH_DIMS)
    configs: list[dict[str, int]] = []
    for dim in selected_dims:
        if dim not in VALID_LAUNCH_DIMS:
            raise ToolError(f"dim must be one of {sorted(VALID_LAUNCH_DIMS)}")
        for rows_per_program, num_warps, num_stages in LAUNCH_PRESETS[dim]:
            configs.append({
                f"rows_per_program_{dim}": rows_per_program,
                f"num_warps_{dim}": num_warps,
                f"num_stages_{dim}": num_stages,
            })
    return configs


def default_joint_launch_overlays() -> list[dict[str, int]]:
    return [
        {},
        {"rows_per_program_2048": 2, "num_warps_2048": 8, "num_stages_2048": 3},
        {"rows_per_program_2048": 2, "num_warps_2048": 4, "num_stages_2048": 3},
        {"rows_per_program_4096": 2, "num_warps_4096": 4, "num_stages_4096": 3},
        {"rows_per_program_4096": 1, "num_warps_4096": 8, "num_stages_4096": 3},
        {
            "rows_per_program_2048": 2,
            "num_warps_2048": 8,
            "num_stages_2048": 3,
            "rows_per_program_4096": 1,
            "num_warps_4096": 8,
            "num_stages_4096": 3,
        },
        {
            "rows_per_program_2048": 2,
            "num_warps_2048": 4,
            "num_stages_2048": 3,
            "rows_per_program_4096": 1,
            "num_warps_4096": 8,
            "num_stages_4096": 3,
        },
        {
            "rows_per_program_2048": 2,
            "num_warps_2048": 4,
            "num_stages_2048": 3,
            "rows_per_program_4096": 2,
            "num_warps_4096": 4,
            "num_stages_4096": 3,
        },
    ]


def detect_silu_variants(text: str) -> dict[str, int]:
    return {name: text.count(expr) for name, expr in SILU_VARIANTS.items() if text.count(expr)}


def render_silu_variant(text: str, variant: str) -> tuple[str, int]:
    if variant not in SILU_VARIANTS:
        raise ToolError(f"silu variant must be one of {sorted(SILU_VARIANTS)}")
    placeholder = "__HARNESSGYM_SILU_EXPR__"
    if placeholder in text:
        raise ToolError("kernel source contains reserved silu replacement placeholder")
    replacement_count = 0
    rendered = text
    for expr in SILU_VARIANTS.values():
        count = rendered.count(expr)
        if count:
            replacement_count += count
            rendered = rendered.replace(expr, placeholder)
    if replacement_count == 0:
        raise ToolError("kernel.py does not contain a recognized SiLU expression to replace")
    return rendered.replace(placeholder, SILU_VARIANTS[variant]), replacement_count


def replace_fragment_outside_defs(text: str, fragment: str, placeholder: str) -> tuple[str, int]:
    parts = []
    count = 0
    cursor = 0
    while True:
        index = text.find(fragment, cursor)
        if index < 0:
            break
        prefix = text[max(0, index - 4):index]
        parts.append(text[cursor:index])
        if prefix == "def ":
            parts.append(fragment)
        else:
            parts.append(placeholder)
            count += 1
        cursor = index + len(fragment)
    if not parts:
        return text, 0
    parts.append(text[cursor:])
    return "".join(parts), count


def validate_silu_approx_variants(raw_variants: Any) -> list[str]:
    if raw_variants is None:
        return sorted(SILU_APPROX_VARIANTS)
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ToolError("variants must be a non-empty list")
    variants = [str(variant) for variant in raw_variants]
    unknown = sorted(set(variants) - set(SILU_APPROX_VARIANTS))
    if unknown:
        raise ToolError(f"unsupported silu approximation variants: {unknown}")
    return variants


def horner_expr(var_name: str, coeffs: list[float]) -> str:
    rendered = f"{coeffs[-1]:.12g}"
    for coeff in reversed(coeffs[:-1]):
        rendered = f"{coeff:.12g} + {var_name} * ({rendered})"
    return rendered


def silu_approx_scalar(gate: float, variant: str) -> float:
    if variant not in SILU_APPROX_VARIANTS:
        raise ToolError(f"unsupported silu approximation variant: {variant}")
    spec = SILU_APPROX_VARIANTS[variant]
    limit = float(spec["range"])
    if gate > limit:
        return gate
    if gate < -limit:
        return 0.0
    t = gate * gate
    num = 0.0
    for coeff in reversed(spec["num"]):
        num = float(coeff) + t * num
    den = 0.0
    for coeff in reversed(spec["den"]):
        den = float(coeff) + t * den
    return gate * (0.5 + gate * num / den)


def silu_exact_scalar(gate: float) -> float:
    return gate / (1.0 + math.exp(-gate))


def fp16_round(value: float) -> float:
    try:
        return float(struct.unpack("<e", struct.pack("<e", float(value)))[0])
    except OverflowError:
        return math.copysign(float("inf"), value)


def approximation_helper_source(variant: str) -> str:
    if variant not in SILU_APPROX_VARIANTS:
        raise ToolError(f"unsupported silu approximation variant: {variant}")
    spec = SILU_APPROX_VARIANTS[variant]
    num = horner_expr("t", [float(value) for value in spec["num"]])
    den = horner_expr("t", [float(value) for value in spec["den"]])
    limit = float(spec["range"])
    return (
        f"{APPROX_HELPER_START}\n"
        "@triton.jit\n"
        "def _harnessgym_silu_approx(gate):\n"
        "    t = gate * gate\n"
        f"    num = {num}\n"
        f"    den = {den}\n"
        "    approx = gate * (0.5 + gate * num / den)\n"
        f"    return tl.where(gate > {limit:.1f}, gate, tl.where(gate < -{limit:.1f}, 0.0, approx))\n"
        f"{APPROX_HELPER_END}\n"
    )


def strip_harnessgym_approx_helper(text: str) -> str:
    start = text.find(APPROX_HELPER_START)
    if start < 0:
        return text
    end = text.find(APPROX_HELPER_END, start)
    if end < 0:
        return text
    end += len(APPROX_HELPER_END)
    while end < len(text) and text[end] in "\r\n":
        end += 1
    return text[:start].rstrip() + "\n\n" + text[end:].lstrip()


def render_silu_approx_variant(text: str, variant: str) -> tuple[str, int]:
    if variant not in SILU_APPROX_VARIANTS:
        raise ToolError(f"silu approximation variant must be one of {sorted(SILU_APPROX_VARIANTS)}")
    placeholder = "__HARNESSGYM_SILU_APPROX_EXPR__"
    if placeholder in text:
        raise ToolError("kernel source contains reserved silu approximation placeholder")
    rendered = strip_harnessgym_approx_helper(text)
    if "import triton\n" not in rendered and "import triton.language as tl" in rendered:
        rendered = rendered.replace("import triton.language as tl\n", "import triton\nimport triton.language as tl\n", 1)
    replacement_count = 0
    fragments = list(SILU_VARIANTS.values()) + [
        "_silu_rational_m2n2(gate)",
        "_silu_rational_m3n2(gate)",
        "_silu_rational_m3n3(gate)",
        "_harnessgym_silu_approx(gate)",
    ]
    for fragment in fragments:
        rendered, count = replace_fragment_outside_defs(rendered, fragment, placeholder)
        if count:
            replacement_count += count
    if replacement_count == 0:
        raise ToolError("kernel.py does not contain a recognized exact or approximate SiLU expression to replace")
    insertion = approximation_helper_source(variant)
    marker = "\n\n@triton.jit\n"
    index = rendered.find(marker)
    if index < 0:
        lines = rendered.splitlines(keepends=True)
        insert_at = 0
        while insert_at < len(lines) and (lines[insert_at].startswith("import ") or lines[insert_at].startswith("from ")):
            insert_at += 1
        rendered = "".join(lines[:insert_at]).rstrip() + "\n\n" + insertion + "\n" + "".join(lines[insert_at:]).lstrip()
    else:
        rendered = rendered[: index + 2] + insertion + "\n" + rendered[index + 2 :]
    return rendered.replace(placeholder, "_harnessgym_silu_approx(gate)"), replacement_count


def render_silu_exact_any_source(text: str, variant: str) -> tuple[str, int]:
    if variant not in SILU_VARIANTS:
        raise ToolError(f"silu variant must be one of {sorted(SILU_VARIANTS)}")
    placeholder = "__HARNESSGYM_SILU_ANY_EXPR__"
    if placeholder in text:
        raise ToolError("kernel source contains reserved silu replacement placeholder")
    rendered = strip_harnessgym_approx_helper(text)
    replacement_count = 0
    fragments = list(SILU_VARIANTS.values()) + [
        "_silu_rational_m2n2(gate)",
        "_silu_rational_m3n2(gate)",
        "_silu_rational_m3n3(gate)",
        "_harnessgym_silu_approx(gate)",
    ]
    for fragment in fragments:
        rendered, count = replace_fragment_outside_defs(rendered, fragment, placeholder)
        if count:
            replacement_count += count
    if replacement_count == 0:
        raise ToolError("kernel.py does not contain a recognized exact or approximate SiLU expression to replace")
    return rendered.replace(placeholder, SILU_VARIANTS[variant]), replacement_count


def render_joint_source_variant(text: str, variant: str) -> tuple[str, int]:
    if variant == "current":
        return text, 0
    if variant in SILU_VARIANTS:
        return render_silu_exact_any_source(text, variant)
    if variant in SILU_APPROX_VARIANTS:
        return render_silu_approx_variant(text, variant)
    supported = ["current", *sorted(SILU_VARIANTS), *sorted(SILU_APPROX_VARIANTS)]
    raise ToolError(f"source variant must be one of {supported}")


def rmsnorm_row(xs: list[float], gates: list[float], weights: list[float], variant: str) -> list[float]:
    mean_sq = sum(x * x for x in xs) / len(xs)
    inv = 1.0 / math.sqrt(mean_sq + 1.0e-5)
    out = []
    for x, gate, weight in zip(xs, gates, weights):
        exact = variant == "exact"
        silu_gate = silu_exact_scalar(gate) if exact else silu_approx_scalar(gate, variant)
        out.append(fp16_round(x * inv * weight * silu_gate))
    return out


def approximation_probe_for_variant(
    workspace: Path,
    variant: str,
    mode: str,
    sample_rows: int,
    tolerance: float,
) -> dict[str, Any]:
    cases_by_mode = parse_cases(workspace)
    selected_modes = ["dev", "final"] if mode == "all" else [validate_mode(mode)]
    toy_x = [
        [0.0, 1.0, -2.0, 3.0, 0.5, -0.75, 2.5, -3.5],
        [1.25, -1.5, 0.25, -0.5, 4.0, -4.0, 0.125, -0.25],
    ]
    toy_gate = [
        [-10.0, -8.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0],
        [8.0, 7.0, 5.0, 1.0, -1.0, -5.0, -7.0, -8.0],
    ]
    toy_weight = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4]
    toy_diffs = []
    for xs, gates in zip(toy_x, toy_gate):
        baseline = rmsnorm_row(xs, gates, toy_weight, "exact")
        candidate = rmsnorm_row(xs, gates, toy_weight, variant)
        toy_diffs.extend(abs(x - y) for x, y in zip(baseline, candidate))

    case_results = []
    all_diffs = list(toy_diffs)
    for selected_mode in selected_modes:
        for case in cases_by_mode.get(selected_mode, []):
            rows = min(max(1, sample_rows), int(case.get("rows", 1)))
            dim = int(case.get("dim", 1))
            rng = random.Random(int(case.get("seed", 0)))
            weights = [0.4 + rng.random() for _ in range(dim)]
            diffs = []
            for _ in range(rows):
                xs = [rng.gauss(0.0, 1.0) for _ in range(dim)]
                gates = [rng.gauss(0.0, 1.0) for _ in range(dim)]
                baseline = rmsnorm_row(xs, gates, weights, "exact")
                candidate = rmsnorm_row(xs, gates, weights, variant)
                diffs.extend(abs(x - y) for x, y in zip(baseline, candidate))
            max_abs = max(diffs) if diffs else 0.0
            all_diffs.extend(diffs)
            case_results.append({
                "mode": selected_mode,
                "name": case.get("name"),
                "rows_sampled": rows,
                "rows_total": case.get("rows"),
                "dim": dim,
                "seed": case.get("seed"),
                "max_abs": max_abs,
                "passed": max_abs <= tolerance,
            })

    grid = [(-10.0 + 20.0 * i / 2000.0) for i in range(2001)]
    silu_diffs = [abs(silu_exact_scalar(gate) - silu_approx_scalar(gate, variant)) for gate in grid]
    max_abs = max(all_diffs) if all_diffs else 0.0
    return {
        "variant": variant,
        "status": "passed" if max_abs <= tolerance else "failed",
        "tolerance": tolerance,
        "max_abs": max_abs,
        "toy": {
            "rows": len(toy_x),
            "dim": len(toy_x[0]),
            "max_abs": max(toy_diffs) if toy_diffs else 0.0,
        },
        "shape_proxy": {
            "mode": mode,
            "sample_rows": sample_rows,
            "cases": case_results,
        },
        "silu_abs_error_grid": {
            "range": [-10.0, 10.0],
            "points": len(grid),
            "max_abs": max(silu_diffs),
        },
        "notes": SILU_APPROX_VARIANTS[variant]["notes"],
    }


def validate_config_overlay(config: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, raw_value in config.items():
        if key in {"num_warps", "num_stages", "block_size", "rows_per_program"}:
            value = int(raw_value)
            if key == "num_warps" and value not in VALID_WARPS:
                raise ToolError(f"{key} must be one of {sorted(VALID_WARPS)}")
            if key == "num_stages" and value not in VALID_STAGES:
                raise ToolError(f"{key} must be one of {sorted(VALID_STAGES)}")
            if key == "rows_per_program" and value not in VALID_ROWS_PER_PROGRAM:
                raise ToolError(f"{key} must be one of {sorted(VALID_ROWS_PER_PROGRAM)}")
            if key == "block_size" and (value < 0 or (value and (value & (value - 1))) or value > 16384):
                raise ToolError("block_size must be 0 or a positive power of two <= 16384")
            out[key] = value
            continue
        out.update(validate_launch_override({key: raw_value}))
    return out


def parse_json_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ToolError(f"command did not emit JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ToolError("command JSON was not an object")
    return value


def validate_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ToolError(f"mode must be one of {sorted(VALID_MODES)}")
    return mode


def validate_config(config: dict[str, Any]) -> dict[str, int]:
    unknown = set(config) - ALLOWED_CONFIG_KEYS
    if unknown:
        raise ToolError(f"unsupported config keys: {sorted(unknown)}")
    out = {
        "num_warps": int(config.get("num_warps", 1)),
        "num_stages": int(config.get("num_stages", 4)),
        "block_size": int(config.get("block_size", 0)),
        "rows_per_program": int(config.get("rows_per_program", 1)),
    }
    if out["num_warps"] not in VALID_WARPS:
        raise ToolError(f"num_warps must be one of {sorted(VALID_WARPS)}")
    if out["num_stages"] not in VALID_STAGES:
        raise ToolError(f"num_stages must be one of {sorted(VALID_STAGES)}")
    if out["rows_per_program"] not in VALID_ROWS_PER_PROGRAM:
        raise ToolError(f"rows_per_program must be one of {sorted(VALID_ROWS_PER_PROGRAM)}")
    block = out["block_size"]
    if block < 0 or (block and (block & (block - 1))):
        raise ToolError("block_size must be 0 or a positive power of two")
    if block > 16384:
        raise ToolError("block_size must be <= 16384")
    return out


def source_state(workspace: Path) -> dict[str, Any]:
    files = {}
    for name in ["kernel.py", "kernel_config.json", "benchmark.py", "verifier.py", "remote_h100.py"]:
        path = workspace / name
        files[name] = {
            "exists": path.exists(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size if path.exists() else None,
        }
    return files


def run_subprocess(command: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "duration_seconds": time.perf_counter() - started,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def ssh_command(host: str, port: str, key: str | None, remote_command: str, timeout: int) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(port),
    ]
    if key:
        command.extend(["-i", str(Path(key).expanduser())])
    command.extend([host, remote_command])
    return command


def classify_remote_failure(return_code: int, stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if return_code == 42 or "hg_no_nvidia_smi" in combined:
        return "gpu"
    if return_code == 43 or "hg_low_disk" in combined:
        return "disk"
    ssh_markers = [
        "connection refused",
        "could not resolve hostname",
        "permission denied",
        "no route to host",
        "operation timed out",
        "connection timed out",
        "connection closed",
        "host key verification failed",
    ]
    if any(marker in combined for marker in ssh_markers):
        return "ssh"
    return "remote"


class HarnessTools:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    def inspect_context(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        activation_path = self.workspace / ".harnessgym/activation.json"
        result_path = latest_result_json(self.workspace)
        baseline_path = next(iter(sorted((self.workspace / ".harnessgym/runs").glob("*/baseline/baseline.stdout.txt"))), None)
        baseline_json = None
        if baseline_path and baseline_path.exists():
            try:
                baseline_json = parse_json_stdout(baseline_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                baseline_json = {"parse_error": str(exc), "path": str(baseline_path)}
        best_manifest_path = next(iter(sorted((self.workspace / ".harnessgym/runs").glob("*/checkpoints/best_manifest.json"))), None)
        return {
            "workspace": str(self.workspace),
            "activation": read_json(activation_path) if activation_path.exists() else None,
            "cases": parse_cases(self.workspace),
            "source": source_state(self.workspace),
            "latest_result_path": str(result_path) if result_path else None,
            "latest_result": read_json(result_path) if result_path and result_path.exists() else None,
            "baseline": baseline_json,
            "best_manifest": read_json(best_manifest_path) if best_manifest_path and best_manifest_path.exists() else None,
            "history_path": str(self.workspace / HISTORY_REL),
        }

    def run_objective(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "dev")))
        verifier = bool(args.get("verifier", False))
        warmup = int(args.get("warmup", 20))
        repeats = int(args.get("repeats", 50))
        timeout = int(args.get("timeout_seconds", 240))
        remote = args.get("remote", "auto")
        if warmup < 0 or repeats <= 0:
            raise ToolError("warmup must be >= 0 and repeats must be > 0")
        if remote not in {"auto", True, False, "true", "false"}:
            raise ToolError("remote must be auto, true, or false")
        use_remote = remote in {True, "true"} or (remote == "auto" and bool(os.environ.get("HARNESSGYM_GPU_HOST")))
        script = "verifier.py" if verifier else "benchmark.py"
        inner = ["python3", script, "--json", "--mode", mode, "--warmup", str(warmup), "--repeats", str(repeats)]
        if use_remote:
            workspace_tag = str(args.get("workspace_tag") or f"h100_rmsnorm_{mode}_{uuid.uuid4().hex[:8]}")
            command = ["python3", "remote_h100.py", "--workspace", workspace_tag, "--", *inner]
        else:
            command = inner
        completed = run_subprocess(command, self.workspace, timeout)
        parsed = None
        parse_error = None
        try:
            parsed = parse_json_stdout(completed["stdout"])
        except Exception as exc:  # noqa: BLE001
            parse_error = str(exc)
        record = {
            "timestamp": time.time(),
            "tool": "run_objective",
            "mode": mode,
            "verifier": verifier,
            "remote": use_remote,
            "config": read_json(self.workspace / "kernel_config.json") if (self.workspace / "kernel_config.json").exists() else None,
            "source": source_state(self.workspace),
            "return_code": completed["return_code"],
            "duration_seconds": completed["duration_seconds"],
            "json": parsed,
            "parse_error": parse_error,
            "command": command,
        }
        append_jsonl(self.workspace / HISTORY_REL, record)
        return {
            "status": "passed" if completed["return_code"] == 0 and parsed and parsed.get("status") == "passed" else "failed",
            "command": command,
            "return_code": completed["return_code"],
            "json": parsed,
            "parse_error": parse_error,
            "history_path": str(self.workspace / HISTORY_REL),
            "stderr_tail": completed["stderr"][-2000:],
        }

    def remote_health_check(self, args: dict[str, Any]) -> dict[str, Any]:
        timeout = int(args.get("timeout_seconds", 8))
        min_free_mb = int(args.get("min_free_mb", 256))
        if timeout <= 0 or timeout > 60:
            raise ToolError("timeout_seconds must be in the range 1..60")
        if min_free_mb < 0:
            raise ToolError("min_free_mb must be >= 0")

        host = str(args.get("host") or os.environ.get("HARNESSGYM_GPU_HOST") or "")
        port = str(args.get("port") or os.environ.get("HARNESSGYM_GPU_PORT", "22"))
        key = args.get("key") or os.environ.get("HARNESSGYM_GPU_KEY")
        remote_root = str(args.get("remote_root") or os.environ.get("HARNESSGYM_GPU_REMOTE_ROOT", "/root/harnessgym_remote"))
        require_remote = bool(args.get("require_remote", False))
        require_gpu = bool(args.get("require_gpu", True))
        dry_run = bool(args.get("dry_run", False))

        local_cuda = None
        if bool(args.get("check_local_cuda", True)) and not host:
            try:
                local = run_subprocess(
                    ["python3", "-c", "import torch; print(bool(torch.cuda.is_available()))"],
                    self.workspace,
                    timeout,
                )
                local_cuda = local["return_code"] == 0 and local["stdout"].strip().splitlines()[-1:] == ["True"]
            except Exception:  # noqa: BLE001
                local_cuda = False

        if not host:
            status = "failed" if require_remote or (require_gpu and not local_cuda) else "passed"
            return {
                "status": status,
                "mode": "local",
                "failure_stage": None if status == "passed" else "configuration",
                "host_configured": False,
                "local_cuda_available": local_cuda,
                "message": (
                    "No HARNESSGYM_GPU_HOST configured and local CUDA is unavailable"
                    if status == "failed"
                    else "No remote host configured; local CUDA is available or not required"
                ),
            }

        min_free_kb = min_free_mb * 1024
        gpu_check = (
            "if command -v nvidia-smi >/dev/null 2>&1; then "
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
            "else echo HG_NO_NVIDIA_SMI >&2; exit 42; fi"
            if require_gpu
            else (
                "if command -v nvidia-smi >/dev/null 2>&1; then "
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; "
                "else echo HG_NO_NVIDIA_SMI; fi"
            )
        )
        remote_script = (
            "set -eu; "
            "echo HG_REMOTE_OK; "
            f"mkdir -p {shlex.quote(remote_root)}; "
            f"FREE_KB=$(df -Pk {shlex.quote(remote_root)} | awk 'NR==2 {{print $4}}'); "
            "echo HG_FREE_KB=${FREE_KB:-0}; "
            f"if [ \"${{FREE_KB:-0}}\" -lt {min_free_kb} ]; then echo HG_LOW_DISK >&2; exit 43; fi; "
            f"{gpu_check}"
        )
        key_arg = str(Path(str(key)).expanduser()) if key else None
        command = ssh_command(host, port, key_arg, remote_script, timeout)
        redacted_command = ["<key>" if key_arg and part == key_arg else part for part in command]
        if dry_run:
            return {
                "status": "dry_run",
                "mode": "remote",
                "host": host,
                "port": port,
                "key_configured": bool(key),
                "remote_root": remote_root,
                "min_free_mb": min_free_mb,
                "require_gpu": require_gpu,
                "command": redacted_command,
            }

        try:
            completed = run_subprocess(command, self.workspace, timeout + 5)
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "failed",
                "mode": "remote",
                "failure_stage": "ssh",
                "host": host,
                "port": port,
                "key_configured": bool(key),
                "return_code": None,
                "command": redacted_command,
                "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                "message": f"SSH preflight timed out after {timeout + 5} seconds",
            }

        stdout = str(completed["stdout"])
        stderr = str(completed["stderr"])
        failure_stage = classify_remote_failure(int(completed["return_code"]), stdout, stderr)
        free_kb = None
        gpu_lines = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("HG_FREE_KB="):
                try:
                    free_kb = int(stripped.split("=", 1)[1])
                except ValueError:
                    free_kb = None
            elif stripped and not stripped.startswith("HG_REMOTE_OK"):
                gpu_lines.append(stripped)
        passed = completed["return_code"] == 0 and (not require_gpu or bool(gpu_lines))
        return {
            "status": "passed" if passed else "failed",
            "mode": "remote",
            "failure_stage": None if passed else failure_stage,
            "host": host,
            "port": port,
            "key_configured": bool(key),
            "remote_root": remote_root,
            "min_free_mb": min_free_mb,
            "free_mb": None if free_kb is None else free_kb / 1024.0,
            "gpu": gpu_lines,
            "return_code": completed["return_code"],
            "duration_seconds": completed["duration_seconds"],
            "command": redacted_command,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }

    def sweep_kernel_config(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "dev")))
        verifier = bool(args.get("verifier", False))
        warmup = int(args.get("warmup", 5))
        repeats = int(args.get("repeats", 15))
        keep_best = bool(args.get("keep_best", False))
        configs = args.get("configs")
        if configs is None:
            warps = args.get("num_warps", [1, 2, 4, 8, 16, 32])
            stages = args.get("num_stages", [1, 2, 3, 4, 5])
            blocks = args.get("block_size", [0])
            configs = [
                {"num_warps": int(w), "num_stages": int(s), "block_size": int(b)}
                for w in warps
                for s in stages
                for b in blocks
            ]
        if not isinstance(configs, list) or not configs:
            raise ToolError("configs must be a non-empty list")
        if len(configs) > int(args.get("max_configs", 120)):
            raise ToolError("too many configs for one rollback-safe sweep")
        config_path = self.workspace / "kernel_config.json"
        original_exists = config_path.exists()
        original_text = config_path.read_text(encoding="utf-8") if original_exists else None
        best: dict[str, Any] | None = None
        results = []
        try:
            for raw_config in configs:
                if not isinstance(raw_config, dict):
                    raise ToolError("each config must be an object")
                config = validate_config(raw_config)
                write_json(config_path, config)
                run = self.run_objective({
                    "mode": mode,
                    "verifier": verifier,
                    "warmup": warmup,
                    "repeats": repeats,
                    "timeout_seconds": int(args.get("timeout_seconds", 240)),
                    "remote": args.get("remote", "auto"),
                })
                score = None
                passed = False
                if run.get("json"):
                    score = run["json"].get("best_us")
                    passed = run["json"].get("status") == "passed"
                item = {
                    "config": config,
                    "status": run.get("status"),
                    "passed": passed,
                    "best_us": score,
                    "return_code": run.get("return_code"),
                }
                results.append(item)
                if passed and isinstance(score, (int, float)) and (best is None or float(score) < float(best["best_us"])):
                    best = item
        finally:
            if keep_best and best is not None:
                write_json(config_path, best["config"])
            elif original_exists and original_text is not None:
                config_path.write_text(original_text, encoding="utf-8")
            elif config_path.exists():
                config_path.unlink()
        return {
            "mode": mode,
            "verifier": verifier,
            "candidate_count": len(results),
            "best": best,
            "kept_best": keep_best and best is not None,
            "restored_original": not (keep_best and best is not None),
            "results": results,
            "history_path": str(self.workspace / HISTORY_REL),
        }

    def restore_best_checkpoint(self, args: dict[str, Any]) -> dict[str, Any]:
        manifest = best_manifest_json(self.workspace)
        if manifest is None:
            raise ToolError("no best checkpoint manifest found")
        checkpoint = Path(str(manifest.get("checkpoint_path", "")))
        if not checkpoint.is_absolute():
            checkpoint = self.workspace / checkpoint
        if not checkpoint.exists():
            raise ToolError(f"best checkpoint path does not exist: {checkpoint}")

        files = args.get("files", sorted(RESTORE_ALLOWLIST))
        if not isinstance(files, list) or not files:
            raise ToolError("files must be a non-empty list")
        restored = []
        for raw_name in files:
            name = str(raw_name)
            if name not in RESTORE_ALLOWLIST:
                raise ToolError(f"restore file must be one of {sorted(RESTORE_ALLOWLIST)}")
            source = checkpoint / name
            target = self.workspace / name
            if not source.exists():
                raise ToolError(f"checkpoint is missing {name}")
            shutil.copy2(source, target)
            restored.append(name)
        return {
            "status": "restored",
            "checkpoint_path": str(checkpoint),
            "score": manifest.get("score"),
            "files": restored,
        }

    def guarded_final_verify(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "final")))
        verifier = bool(args.get("verifier", True))
        warmup = int(args.get("warmup", 20))
        repeats = int(args.get("repeats", 50))
        timeout = int(args.get("timeout_seconds", 240))
        restore_on_regression = bool(args.get("restore_on_regression", True))
        dry_run = bool(args.get("dry_run", False))
        manifest = best_manifest_json(self.workspace)
        threshold = args.get("max_score", manifest.get("score") if manifest else None)
        threshold_value = float(threshold) if isinstance(threshold, (int, float)) else None
        restore_files = args.get("restore_files", sorted(RESTORE_ALLOWLIST))
        if dry_run:
            return {
                "status": "dry_run",
                "mode": mode,
                "verifier": verifier,
                "restore_on_regression": restore_on_regression,
                "threshold_score": threshold_value,
                "restore_files": restore_files,
            }

        run = self.run_objective({
            "mode": mode,
            "verifier": verifier,
            "warmup": warmup,
            "repeats": repeats,
            "timeout_seconds": timeout,
            "remote": args.get("remote", "auto"),
        })
        parsed = run.get("json") or {}
        score = parsed.get("best_us")
        passed = run.get("status") == "passed" and parsed.get("status") == "passed"
        regressed = (
            not passed
            or (
                threshold_value is not None
                and isinstance(score, (int, float))
                and float(score) > threshold_value
            )
        )
        restored = None
        if regressed and restore_on_regression:
            restored = self.restore_best_checkpoint({"files": restore_files})
        return {
            "status": "passed" if passed and not regressed else "regressed",
            "score": float(score) if isinstance(score, (int, float)) else None,
            "threshold_score": threshold_value,
            "restored": restored,
            "run": run,
        }

    def sweep_launch_overrides(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "dev")))
        verifier = bool(args.get("verifier", mode == "final"))
        warmup = int(args.get("warmup", 8))
        repeats = int(args.get("repeats", 20))
        keep_best = bool(args.get("keep_best", False))
        dry_run = bool(args.get("dry_run", False))
        max_configs = int(args.get("max_configs", 48))
        dims_arg = args.get("dims")
        dims = [int(dim) for dim in dims_arg] if isinstance(dims_arg, list) else None
        raw_configs = args.get("configs")
        if raw_configs is None:
            configs = generated_launch_overrides(dims)
        else:
            if not isinstance(raw_configs, list) or not raw_configs:
                raise ToolError("configs must be a non-empty list")
            configs = [validate_launch_override(config) for config in raw_configs]
        if len(configs) > max_configs:
            raise ToolError("too many launch override configs for one rollback-safe sweep")

        source_support = source_supports_launch_overrides(self.workspace)
        if dry_run:
            return {
                "status": "dry_run",
                "mode": mode,
                "verifier": verifier,
                "candidate_count": len(configs),
                "source_supports_launch_overrides": source_support,
                "configs": configs,
            }
        if not source_support:
            raise ToolError(
                "kernel.py does not expose rows_per_program_<dim>, num_warps_<dim>, "
                "and num_stages_<dim> config hooks; add the tunable launch hook before sweeping"
            )

        config_path = self.workspace / "kernel_config.json"
        original_exists = config_path.exists()
        original_text = config_path.read_text(encoding="utf-8") if original_exists else None
        base_config = read_json(config_path) if original_exists else {}
        best: dict[str, Any] | None = None
        results = []
        try:
            for overrides in configs:
                merged = {**base_config, **overrides}
                write_json(config_path, merged)
                run = self.run_objective({
                    "mode": mode,
                    "verifier": verifier,
                    "warmup": warmup,
                    "repeats": repeats,
                    "timeout_seconds": int(args.get("timeout_seconds", 240)),
                    "remote": args.get("remote", "auto"),
                })
                parsed = run.get("json") or {}
                score = parsed.get("best_us")
                passed = run.get("status") == "passed" and parsed.get("status") == "passed"
                item = {
                    "overrides": overrides,
                    "status": run.get("status"),
                    "passed": passed,
                    "best_us": score,
                    "cases": parsed.get("cases", []),
                }
                results.append(item)
                if passed and isinstance(score, (int, float)) and (best is None or float(score) < float(best["best_us"])):
                    best = item
        finally:
            if keep_best and best is not None:
                write_json(config_path, {**base_config, **best["overrides"]})
            elif original_exists and original_text is not None:
                config_path.write_text(original_text, encoding="utf-8")
            elif config_path.exists():
                config_path.unlink()
        return {
            "mode": mode,
            "verifier": verifier,
            "candidate_count": len(results),
            "source_supports_launch_overrides": source_support,
            "best": best,
            "kept_best": keep_best and best is not None,
            "restored_original": not (keep_best and best is not None),
            "results": results,
            "history_path": str(self.workspace / HISTORY_REL),
        }

    def sweep_silu_variants(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "dev")))
        verifier = bool(args.get("verifier", mode == "final"))
        warmup = int(args.get("warmup", 8))
        repeats = int(args.get("repeats", 20))
        timeout = int(args.get("timeout_seconds", 240))
        dry_run = bool(args.get("dry_run", False))
        keep_best = bool(args.get("keep_best", False))
        raw_variants = args.get("variants", sorted(SILU_VARIANTS))
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ToolError("variants must be a non-empty list")
        variants = [str(variant) for variant in raw_variants]
        unknown_variants = sorted(set(variants) - set(SILU_VARIANTS))
        if unknown_variants:
            raise ToolError(f"unsupported silu variants: {unknown_variants}")

        raw_overlays = args.get("config_overlays", [{}])
        if not isinstance(raw_overlays, list) or not raw_overlays:
            raise ToolError("config_overlays must be a non-empty list")
        overlays = []
        for raw_overlay in raw_overlays:
            if not isinstance(raw_overlay, dict):
                raise ToolError("each config overlay must be an object")
            overlays.append(validate_config_overlay(raw_overlay))
        max_candidates = int(args.get("max_candidates", 24))
        if len(variants) * len(overlays) > max_candidates:
            raise ToolError("too many silu/config candidates for one rollback-safe sweep")

        kernel_path = self.workspace / "kernel.py"
        config_path = self.workspace / "kernel_config.json"
        if not kernel_path.exists():
            raise ToolError("kernel.py does not exist")
        original_kernel = kernel_path.read_text(encoding="utf-8")
        detected = detect_silu_variants(original_kernel)
        if not detected:
            raise ToolError("kernel.py does not contain a recognized SiLU expression to replace")
        original_config_exists = config_path.exists()
        original_config_text = config_path.read_text(encoding="utf-8") if original_config_exists else None
        base_config = read_json(config_path) if original_config_exists else {}
        plan = [
            {"variant": variant, "config_overlay": overlay}
            for variant in variants
            for overlay in overlays
        ]
        if dry_run:
            return {
                "status": "dry_run",
                "mode": mode,
                "verifier": verifier,
                "source_detected": detected,
                "candidate_count": len(plan),
                "candidates": plan,
                "silu_variants": SILU_VARIANTS,
            }

        best: dict[str, Any] | None = None
        best_kernel_text: str | None = None
        best_config_text: str | None = None
        results = []
        try:
            for variant in variants:
                variant_text, replacements = render_silu_variant(original_kernel, variant)
                kernel_path.write_text(variant_text, encoding="utf-8")
                for overlay in overlays:
                    merged_config = {**base_config, **overlay}
                    if merged_config:
                        write_json(config_path, merged_config)
                    run = self.run_objective({
                        "mode": mode,
                        "verifier": verifier,
                        "warmup": warmup,
                        "repeats": repeats,
                        "timeout_seconds": timeout,
                        "remote": args.get("remote", "auto"),
                    })
                    parsed = run.get("json") or {}
                    score = parsed.get("best_us")
                    passed = run.get("status") == "passed" and parsed.get("status") == "passed"
                    item = {
                        "variant": variant,
                        "config_overlay": overlay,
                        "source_replacements": replacements,
                        "status": run.get("status"),
                        "passed": passed,
                        "best_us": score,
                        "cases": parsed.get("cases", []),
                    }
                    results.append(item)
                    if passed and isinstance(score, (int, float)) and (best is None or float(score) < float(best["best_us"])):
                        best = item
                        best_kernel_text = variant_text
                        best_config_text = json.dumps(merged_config, indent=2, sort_keys=True) + "\n"
        finally:
            if keep_best and best is not None and best_kernel_text is not None:
                kernel_path.write_text(best_kernel_text, encoding="utf-8")
                if best_config_text is not None:
                    config_path.write_text(best_config_text, encoding="utf-8")
            else:
                kernel_path.write_text(original_kernel, encoding="utf-8")
                if original_config_exists and original_config_text is not None:
                    config_path.write_text(original_config_text, encoding="utf-8")
                elif config_path.exists():
                    config_path.unlink()
        return {
            "mode": mode,
            "verifier": verifier,
            "candidate_count": len(results),
            "source_detected": detected,
            "best": best,
            "kept_best": keep_best and best is not None,
            "restored_original": not (keep_best and best is not None),
            "results": results,
            "history_path": str(self.workspace / HISTORY_REL),
        }

    def repeat_objective(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "final")))
        verifier = bool(args.get("verifier", mode == "final"))
        warmup = int(args.get("warmup", 20))
        repeats = int(args.get("repeats", 50))
        runs = int(args.get("runs", 3))
        timeout = int(args.get("timeout_seconds", 240))
        max_runs = int(args.get("max_runs", 10))
        dry_run = bool(args.get("dry_run", False))
        if runs <= 0 or runs > max_runs:
            raise ToolError(f"runs must be between 1 and {max_runs}")
        if dry_run:
            return {
                "status": "dry_run",
                "mode": mode,
                "verifier": verifier,
                "runs": runs,
                "warmup": warmup,
                "repeats": repeats,
                "objective": "minimize total best_us; use median_score to resist one lucky timing run",
            }

        run_items = []
        scores = []
        case_scores: dict[str, dict[str, Any]] = {}
        for index in range(runs):
            run = self.run_objective({
                "mode": mode,
                "verifier": verifier,
                "warmup": warmup,
                "repeats": repeats,
                "timeout_seconds": timeout,
                "remote": args.get("remote", "auto"),
                "workspace_tag": f"h100_rmsnorm_repeat_{mode}_{index}_{uuid.uuid4().hex[:6]}",
            })
            parsed = run.get("json") or {}
            score = parsed.get("best_us")
            passed = run.get("status") == "passed" and parsed.get("status") == "passed"
            if passed and isinstance(score, (int, float)):
                scores.append(float(score))
            for case in parsed.get("cases", []):
                name = str(case.get("name"))
                best_us = case.get("best_us")
                if not isinstance(best_us, (int, float)):
                    continue
                bucket = case_scores.setdefault(name, {
                    "name": name,
                    "dim": case.get("dim"),
                    "rows": case.get("rows"),
                    "samples": [],
                })
                bucket["samples"].append(float(best_us))
            run_items.append({
                "index": index,
                "status": run.get("status"),
                "passed": passed,
                "best_us": score,
                "return_code": run.get("return_code"),
            })

        case_summary = []
        for bucket in case_scores.values():
            samples = bucket.pop("samples")
            bucket.update({
                "min_best_us": min(samples),
                "median_best_us": statistics.median(samples),
                "max_best_us": max(samples),
                "sample_count": len(samples),
            })
            case_summary.append(bucket)
        case_summary.sort(key=lambda item: (int(item.get("dim") or 0), str(item.get("name"))))

        restored = None
        threshold = args.get("max_score")
        if threshold is None:
            manifest = best_manifest_json(self.workspace)
            threshold = manifest.get("score") if manifest else None
        threshold_value = float(threshold) if isinstance(threshold, (int, float)) else None
        if bool(args.get("restore_on_regression", False)) and (
            not scores or (threshold_value is not None and min(scores) > threshold_value)
        ):
            restored = self.restore_best_checkpoint({"files": sorted(RESTORE_ALLOWLIST)})

        return {
            "status": "passed" if len(scores) == runs else "failed",
            "mode": mode,
            "verifier": verifier,
            "runs": runs,
            "score_count": len(scores),
            "min_score": min(scores) if scores else None,
            "median_score": statistics.median(scores) if scores else None,
            "mean_score": statistics.mean(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "score_spread": (max(scores) - min(scores)) if scores else None,
            "threshold_score": threshold_value,
            "restored": restored,
            "cases": case_summary,
            "results": run_items,
            "history_path": str(self.workspace / HISTORY_REL),
        }

    def recommend_next_experiments(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "final")))
        top_n = int(args.get("top_n", 8))
        ranked = self.rank_history({"mode": mode, "top_n": top_n})
        context = self.inspect_context({})
        cases = context.get("cases", {})
        final_cases = cases.get("final", [])
        dev_cases = cases.get("dev", [])
        per_case: dict[str, dict[str, Any]] = {}
        for record in ranked.get("top", []):
            for case in record.get("cases", []):
                name = str(case.get("name"))
                best_us = case.get("best_us")
                if not isinstance(best_us, (int, float)):
                    continue
                current = per_case.get(name)
                if current is None or float(best_us) < float(current["best_us"]):
                    per_case[name] = {
                        "name": name,
                        "dim": case.get("dim"),
                        "rows": case.get("rows"),
                        "best_us": float(best_us),
                        "record_score": record.get("score"),
                        "config": record.get("config"),
                    }
        recommendations = [
            "Use joint_source_launch_search to cross exact/rational SiLU source variants with combined 2048/4096 launch overlays before manual source edits.",
            "Run repeat_objective on the current best final candidate before trusting one low-latency verifier sample.",
            "Use probe_silu_approximations before benchmarking approximate SiLU math; final tolerance can reject variants that look safe on a small toy probe.",
            "Use sweep_silu_approximations rather than manual rational SiLU edits so slow approximate candidates are restored automatically.",
            "Use sweep_silu_variants with exp, sigmoid, and exp2 across the best 8192 launch overlay before manual kernel edits.",
        ]
        if not source_supports_launch_overrides(self.workspace):
            recommendations.insert(0, "Add per-dim launch hooks before running sweep_launch_overrides; otherwise launch sweeps cannot mutate anything.")
        if len(final_cases) > len(dev_cases):
            recommendations.append("Treat dev as a compile/filter pass only; final has held-out rows and the 8192 case that dominated the baseline.")
        return {
            "status": "ready",
            "mode": mode,
            "objective": "minimize total best_us",
            "history_path": ranked.get("history_path"),
            "ranked_record_count": ranked.get("record_count"),
            "best_record": ranked.get("best"),
            "per_case_best": sorted(per_case.values(), key=lambda item: (int(item.get("dim") or 0), str(item.get("name")))),
            "shape_assumptions": {
                "dev": dev_cases,
                "final": final_cases,
                "final_is_held_out": True,
            },
            "overfitting_risks": [
                "Dev lacks final row counts and the 8192-wide case.",
                "Single verifier samples can select lucky timings; aggregate repeated final samples before keeping a candidate.",
            ],
            "recommended_commands": [
                {
                    "tool": "joint_source_launch_search",
                    "arguments": {
                        "mode": "dev",
                        "verifier": False,
                        "source_variants": ["current", "sigmoid", "exp2", "rational_m3n2"],
                        "warmup": 8,
                        "repeats": 20,
                        "confirm_top_n": 2,
                        "confirm_runs": 3,
                    },
                },
                {
                    "tool": "probe_silu_approximations",
                    "arguments": {
                        "mode": "all",
                        "variants": ["rational_m2n2", "rational_m3n2", "rational_m3n3"],
                        "sample_rows": 4,
                        "tolerance": 0.0075,
                    },
                },
                {
                    "tool": "sweep_silu_approximations",
                    "arguments": {
                        "mode": "final",
                        "verifier": True,
                        "variants": ["rational_m3n2", "rational_m3n3"],
                        "warmup": 12,
                        "repeats": 30,
                        "keep_best": False,
                    },
                },
                {
                    "tool": "sweep_silu_variants",
                    "arguments": {
                        "mode": "final",
                        "verifier": True,
                        "variants": ["exp", "sigmoid", "exp2"],
                        "warmup": 12,
                        "repeats": 30,
                    },
                },
                {
                    "tool": "repeat_objective",
                    "arguments": {
                        "mode": "final",
                        "verifier": True,
                        "runs": 3,
                        "restore_on_regression": True,
                    },
                },
            ],
            "recommendations": recommendations,
        }

    def probe_silu_approximations(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = str(args.get("mode", "all"))
        if mode not in {"dev", "final", "all"}:
            raise ToolError("mode must be one of ['all', 'dev', 'final']")
        variants = validate_silu_approx_variants(args.get("variants"))
        sample_rows = int(args.get("sample_rows", 4))
        tolerance = float(args.get("tolerance", 7.5e-3))
        if sample_rows <= 0 or sample_rows > 64:
            raise ToolError("sample_rows must be between 1 and 64")
        if tolerance <= 0:
            raise ToolError("tolerance must be positive")
        results = [
            approximation_probe_for_variant(self.workspace, variant, mode, sample_rows, tolerance)
            for variant in variants
        ]
        passed = [result["variant"] for result in results if result["status"] == "passed"]
        return {
            "status": "passed" if len(passed) == len(results) else "failed",
            "mode": mode,
            "objective": "numerically prefilter approximate SiLU formulas before benchmark sweeps",
            "tolerance": tolerance,
            "sample_rows": sample_rows,
            "variants": variants,
            "passed_variants": passed,
            "failed_variants": [result["variant"] for result in results if result["status"] != "passed"],
            "shape_assumptions": parse_cases(self.workspace),
            "results": results,
        }

    def sweep_silu_approximations(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "dev")))
        verifier = bool(args.get("verifier", mode == "final"))
        warmup = int(args.get("warmup", 8))
        repeats = int(args.get("repeats", 20))
        timeout = int(args.get("timeout_seconds", 240))
        dry_run = bool(args.get("dry_run", False))
        keep_best = bool(args.get("keep_best", False))
        variants = validate_silu_approx_variants(args.get("variants"))
        numerical_mode = str(args.get("numerical_mode", "all"))
        if numerical_mode not in {"dev", "final", "all"}:
            raise ToolError("numerical_mode must be one of ['all', 'dev', 'final']")
        sample_rows = int(args.get("sample_rows", 4))
        tolerance = float(args.get("tolerance", 7.5e-3))
        if sample_rows <= 0 or sample_rows > 64:
            raise ToolError("sample_rows must be between 1 and 64")
        if tolerance <= 0:
            raise ToolError("tolerance must be positive")

        raw_overlays = args.get("config_overlays", [{}])
        if not isinstance(raw_overlays, list) or not raw_overlays:
            raise ToolError("config_overlays must be a non-empty list")
        overlays = []
        for raw_overlay in raw_overlays:
            if not isinstance(raw_overlay, dict):
                raise ToolError("each config overlay must be an object")
            overlays.append(validate_config_overlay(raw_overlay))
        max_candidates = int(args.get("max_candidates", 24))
        if len(variants) * len(overlays) > max_candidates:
            raise ToolError("too many approximation/config candidates for one rollback-safe sweep")

        kernel_path = self.workspace / "kernel.py"
        config_path = self.workspace / "kernel_config.json"
        if not kernel_path.exists():
            raise ToolError("kernel.py does not exist")
        original_kernel = kernel_path.read_text(encoding="utf-8")
        # Validate before mutating so invalid sources do not disturb the workspace.
        for variant in variants:
            render_silu_approx_variant(original_kernel, variant)
        original_config_exists = config_path.exists()
        original_config_text = config_path.read_text(encoding="utf-8") if original_config_exists else None
        base_config = read_json(config_path) if original_config_exists else {}
        plan = [
            {"variant": variant, "config_overlay": overlay}
            for variant in variants
            for overlay in overlays
        ]
        if dry_run:
            probes = [
                approximation_probe_for_variant(self.workspace, variant, numerical_mode, sample_rows, tolerance)
                for variant in variants
            ]
            return {
                "status": "dry_run",
                "mode": mode,
                "verifier": verifier,
                "numerical_mode": numerical_mode,
                "candidate_count": len(plan),
                "candidates": plan,
                "approximation_variants": SILU_APPROX_VARIANTS,
                "numerical_probes": probes,
            }

        best: dict[str, Any] | None = None
        best_kernel_text: str | None = None
        best_config_text: str | None = None
        results = []
        try:
            for variant in variants:
                probe = approximation_probe_for_variant(self.workspace, variant, numerical_mode, sample_rows, tolerance)
                if probe["status"] != "passed":
                    for overlay in overlays:
                        results.append({
                            "variant": variant,
                            "config_overlay": overlay,
                            "status": "skipped_numerical",
                            "passed": False,
                            "best_us": None,
                            "numerical_probe": probe,
                            "cases": [],
                        })
                    continue
                variant_text, replacements = render_silu_approx_variant(original_kernel, variant)
                kernel_path.write_text(variant_text, encoding="utf-8")
                for overlay in overlays:
                    merged_config = {**base_config, **overlay}
                    if merged_config:
                        write_json(config_path, merged_config)
                    run = self.run_objective({
                        "mode": mode,
                        "verifier": verifier,
                        "warmup": warmup,
                        "repeats": repeats,
                        "timeout_seconds": timeout,
                        "remote": args.get("remote", "auto"),
                    })
                    parsed = run.get("json") or {}
                    score = parsed.get("best_us")
                    passed = run.get("status") == "passed" and parsed.get("status") == "passed"
                    item = {
                        "variant": variant,
                        "config_overlay": overlay,
                        "source_replacements": replacements,
                        "status": run.get("status"),
                        "passed": passed,
                        "best_us": score,
                        "numerical_probe": probe,
                        "cases": parsed.get("cases", []),
                    }
                    results.append(item)
                    if passed and isinstance(score, (int, float)) and (best is None or float(score) < float(best["best_us"])):
                        best = item
                        best_kernel_text = variant_text
                        best_config_text = json.dumps(merged_config, indent=2, sort_keys=True) + "\n"
        finally:
            if keep_best and best is not None and best_kernel_text is not None:
                kernel_path.write_text(best_kernel_text, encoding="utf-8")
                if best_config_text is not None:
                    config_path.write_text(best_config_text, encoding="utf-8")
            else:
                kernel_path.write_text(original_kernel, encoding="utf-8")
                if original_config_exists and original_config_text is not None:
                    config_path.write_text(original_config_text, encoding="utf-8")
                elif config_path.exists():
                    config_path.unlink()
        return {
            "mode": mode,
            "verifier": verifier,
            "numerical_mode": numerical_mode,
            "candidate_count": len(results),
            "best": best,
            "kept_best": keep_best and best is not None,
            "restored_original": not (keep_best and best is not None),
            "results": results,
            "history_path": str(self.workspace / HISTORY_REL),
        }

    def joint_source_launch_search(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = validate_mode(str(args.get("mode", "dev")))
        verifier = bool(args.get("verifier", mode == "final"))
        warmup = int(args.get("warmup", 8))
        repeats = int(args.get("repeats", 20))
        timeout = int(args.get("timeout_seconds", 240))
        dry_run = bool(args.get("dry_run", False))
        keep_best = bool(args.get("keep_best", False))
        raw_variants = args.get("source_variants", ["current", "sigmoid", "exp2", "rational_m3n2"])
        if not isinstance(raw_variants, list) or not raw_variants:
            raise ToolError("source_variants must be a non-empty list")
        source_variants = [str(variant) for variant in raw_variants]
        raw_overlays = args.get("config_overlays", default_joint_launch_overlays())
        if not isinstance(raw_overlays, list) or not raw_overlays:
            raise ToolError("config_overlays must be a non-empty list")
        overlays = []
        for raw_overlay in raw_overlays:
            if not isinstance(raw_overlay, dict):
                raise ToolError("each config overlay must be an object")
            overlays.append(validate_config_overlay(raw_overlay))
        max_candidates = int(args.get("max_candidates", 32))
        if len(source_variants) * len(overlays) > max_candidates:
            raise ToolError("too many source/config candidates for one rollback-safe search")
        confirm_top_n = int(args.get("confirm_top_n", 0))
        confirm_runs = int(args.get("confirm_runs", 3))
        confirm_warmup = int(args.get("confirm_warmup", 20))
        confirm_repeats = int(args.get("confirm_repeats", 50))
        if confirm_top_n < 0:
            raise ToolError("confirm_top_n must be >= 0")
        if confirm_runs <= 0:
            raise ToolError("confirm_runs must be > 0")

        kernel_path = self.workspace / "kernel.py"
        config_path = self.workspace / "kernel_config.json"
        if not kernel_path.exists():
            raise ToolError("kernel.py does not exist")
        original_kernel = kernel_path.read_text(encoding="utf-8")
        source_texts: dict[str, dict[str, Any]] = {}
        for variant in source_variants:
            rendered, replacements = render_joint_source_variant(original_kernel, variant)
            source_texts[variant] = {"text": rendered, "replacements": replacements}
        original_config_exists = config_path.exists()
        original_config_text = config_path.read_text(encoding="utf-8") if original_config_exists else None
        base_config = read_json(config_path) if original_config_exists else {}
        plan = [
            {"source_variant": variant, "config_overlay": overlay}
            for variant in source_variants
            for overlay in overlays
        ]
        if dry_run:
            return {
                "status": "dry_run",
                "mode": mode,
                "verifier": verifier,
                "candidate_count": len(plan),
                "confirm_top_n": confirm_top_n,
                "source_variants": source_variants,
                "config_overlays": overlays,
                "source_supports_launch_overrides": source_supports_launch_overrides(self.workspace),
                "candidates": plan,
                "objective": "minimize total best_us; confirm in final verifier mode to avoid dev-only overfit",
            }

        def write_candidate(candidate: dict[str, Any]) -> tuple[str, str | None]:
            source_variant = str(candidate["source_variant"])
            overlay = dict(candidate["config_overlay"])
            kernel_text = str(source_texts[source_variant]["text"])
            merged_config = {**base_config, **overlay}
            kernel_path.write_text(kernel_text, encoding="utf-8")
            config_text = json.dumps(merged_config, indent=2, sort_keys=True) + "\n" if merged_config else None
            if config_text is not None:
                config_path.write_text(config_text, encoding="utf-8")
            elif config_path.exists():
                config_path.unlink()
            return kernel_text, config_text

        results = []
        initial_best: dict[str, Any] | None = None
        best: dict[str, Any] | None = None
        best_kernel_text: str | None = None
        best_config_text: str | None = None
        confirmation_results = []
        try:
            for candidate in plan:
                kernel_text, config_text = write_candidate(candidate)
                run = self.run_objective({
                    "mode": mode,
                    "verifier": verifier,
                    "warmup": warmup,
                    "repeats": repeats,
                    "timeout_seconds": timeout,
                    "remote": args.get("remote", "auto"),
                })
                parsed = run.get("json") or {}
                score = parsed.get("best_us")
                passed = run.get("status") == "passed" and parsed.get("status") == "passed"
                item = {
                    "source_variant": candidate["source_variant"],
                    "config_overlay": candidate["config_overlay"],
                    "source_replacements": source_texts[str(candidate["source_variant"])]["replacements"],
                    "status": run.get("status"),
                    "passed": passed,
                    "best_us": score,
                    "cases": parsed.get("cases", []),
                }
                results.append(item)
                if passed and isinstance(score, (int, float)) and (
                    initial_best is None or float(score) < float(initial_best["best_us"])
                ):
                    initial_best = item
                    best = item
                    best_kernel_text = kernel_text
                    best_config_text = config_text

            passed_results = [
                item for item in results
                if item.get("passed") and isinstance(item.get("best_us"), (int, float))
            ]
            passed_results.sort(key=lambda item: float(item["best_us"]))
            if confirm_top_n and passed_results:
                for item in passed_results[:confirm_top_n]:
                    candidate = {
                        "source_variant": item["source_variant"],
                        "config_overlay": item["config_overlay"],
                    }
                    kernel_text, config_text = write_candidate(candidate)
                    repeat = self.repeat_objective({
                        "mode": "final",
                        "verifier": True,
                        "runs": confirm_runs,
                        "max_runs": max(confirm_runs, 10),
                        "warmup": confirm_warmup,
                        "repeats": confirm_repeats,
                        "timeout_seconds": timeout,
                        "remote": args.get("remote", "auto"),
                    })
                    item["confirmation"] = {
                        "status": repeat.get("status"),
                        "runs": repeat.get("runs"),
                        "min_score": repeat.get("min_score"),
                        "median_score": repeat.get("median_score"),
                        "max_score": repeat.get("max_score"),
                        "score_spread": repeat.get("score_spread"),
                    }
                    confirmation_results.append({
                        "candidate": {
                            "source_variant": item["source_variant"],
                            "config_overlay": item["config_overlay"],
                            "best_us": item["best_us"],
                        },
                        "repeat": item["confirmation"],
                    })
                    median_score = repeat.get("median_score")
                    best_median = best.get("confirmation", {}).get("median_score") if best else None
                    if repeat.get("status") == "passed" and isinstance(median_score, (int, float)) and (
                        best is None
                        or not isinstance(best_median, (int, float))
                        or float(median_score) < float(best_median)
                    ):
                        best = item
                        best_kernel_text = kernel_text
                        best_config_text = config_text
        finally:
            if keep_best and best is not None and best_kernel_text is not None:
                kernel_path.write_text(best_kernel_text, encoding="utf-8")
                if best_config_text is not None:
                    config_path.write_text(best_config_text, encoding="utf-8")
                elif config_path.exists():
                    config_path.unlink()
            else:
                kernel_path.write_text(original_kernel, encoding="utf-8")
                if original_config_exists and original_config_text is not None:
                    config_path.write_text(original_config_text, encoding="utf-8")
                elif config_path.exists():
                    config_path.unlink()

        return {
            "mode": mode,
            "verifier": verifier,
            "candidate_count": len(results),
            "source_supports_launch_overrides": source_supports_launch_overrides(self.workspace),
            "initial_best": initial_best,
            "best": best,
            "confirmation_results": confirmation_results,
            "kept_best": keep_best and best is not None,
            "restored_original": not (keep_best and best is not None),
            "results": results,
            "history_path": str(self.workspace / HISTORY_REL),
        }

    def rank_history(self, args: dict[str, Any]) -> dict[str, Any]:
        top_n = int(args.get("top_n", 10))
        mode_filter = args.get("mode")
        if mode_filter is not None:
            validate_mode(str(mode_filter))
        path = Path(args.get("history_path", self.workspace / HISTORY_REL))
        if not path.is_absolute():
            path = self.workspace / path
        records = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = rec.get("json") or {}
                if mode_filter and result.get("mode") != mode_filter:
                    continue
                score = result.get("best_us")
                if isinstance(score, (int, float)):
                    records.append({
                        "timestamp": rec.get("timestamp"),
                        "mode": result.get("mode"),
                        "verifier": rec.get("verifier"),
                        "score": float(score),
                        "status": result.get("status"),
                        "config": rec.get("config"),
                        "cases": result.get("cases", []),
                        "command": rec.get("command"),
                    })
        records.sort(key=lambda r: r["score"])
        return {
            "history_path": str(path),
            "record_count": len(records),
            "top": records[:top_n],
            "best": records[0] if records else None,
        }

    def diagnose_source(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        kernel_path = self.workspace / "kernel.py"
        text = kernel_path.read_text(encoding="utf-8") if kernel_path.exists() else ""
        diff = subprocess.run(
            ["git", "diff", "--", "kernel.py", "kernel_config.json"],
            cwd=str(self.workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "source": source_state(self.workspace),
            "patterns": {
                "uses_tl_sigmoid": "tl.sigmoid" in text,
                "uses_exp_silu": "tl.exp(-gate)" in text or "exp(-gate)" in text,
                "uses_exp2_silu": "tl.exp2" in text,
                "silu_variants_detected": detect_silu_variants(text),
                "uses_silu_approximation": "_harnessgym_silu_approx" in text or "_silu_rational_" in text,
                "silu_approximation_markers": {
                    name: (name in text or name.replace("rational_", "_silu_rational_") in text)
                    for name in SILU_APPROX_VARIANTS
                },
                "has_dim_specialization": "dim >=" in text and "num_warps" in text,
                "has_multirow_kernel": "_rows_kernel" in text or "rows_per_program" in text,
                "loads_weight_per_program": "tl.load(weight_ptr" in text,
            },
            "diff_return_code": diff.returncode,
            "diff": diff.stdout[-12000:],
            "diff_stderr": diff.stderr[-2000:],
        }

    def numerical_probe(self, args: dict[str, Any]) -> dict[str, Any]:
        rows = int(args.get("rows", 3))
        dim = int(args.get("dim", 8))
        seed = int(args.get("seed", 1701))
        tolerance = float(args.get("tolerance", 7.5e-3))
        if rows <= 0 or dim <= 0 or rows * dim > 4096:
            raise ToolError("rows and dim must be positive with rows*dim <= 4096")

        def silu_exp(g: float) -> float:
            return g / (1.0 + math.exp(-g))

        def silu_sigmoid(g: float) -> float:
            return g * (1.0 / (1.0 + math.exp(-g)))

        def silu_exp2(g: float) -> float:
            return g / (1.0 + math.pow(2.0, -LOG2_E * g))

        def rmsnorm(xs: list[float], gates: list[float], weights: list[float], variant: str) -> list[float]:
            mean_sq = sum(x * x for x in xs) / len(xs)
            inv = 1.0 / math.sqrt(mean_sq + 1.0e-5)
            out = []
            for x, g, w in zip(xs, gates, weights):
                if variant == "sigmoid":
                    sg = silu_sigmoid(g)
                elif variant == "exp2":
                    sg = silu_exp2(g)
                else:
                    sg = silu_exp(g)
                out.append(x * inv * w * sg)
            return out

        toy_x = [[0.0, 1.0, -2.0, 3.0], [0.5, -0.25, 0.75, -1.25]]
        toy_gate = [[-1.0, 0.0, 1.0, 2.0], [3.0, -3.0, 0.25, -0.5]]
        toy_weight = [0.4, 0.7, 1.0, 1.3]
        toy_diffs = []
        for xs, gs in zip(toy_x, toy_gate):
            baseline = rmsnorm(xs, gs, toy_weight, "exp")
            for variant in ["sigmoid", "exp2"]:
                candidate = rmsnorm(xs, gs, toy_weight, variant)
                toy_diffs.extend(abs(x - y) for x, y in zip(baseline, candidate))

        rng = random.Random(seed)
        weights = [0.4 + rng.random() for _ in range(dim)]
        rand_diffs = []
        for _ in range(rows):
            xs = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            gs = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            baseline = rmsnorm(xs, gs, weights, "exp")
            for variant in ["sigmoid", "exp2"]:
                candidate = rmsnorm(xs, gs, weights, variant)
                rand_diffs.extend(abs(x - y) for x, y in zip(baseline, candidate))
        max_abs = max(toy_diffs + rand_diffs) if toy_diffs or rand_diffs else 0.0
        return {
            "status": "passed" if max_abs <= tolerance else "failed",
            "tolerance": tolerance,
            "max_abs": max_abs,
            "known_toy": {
                "rows": 2,
                "dim": 4,
                "max_abs": max(toy_diffs),
            },
            "fixed_seed_random": {
                "rows": rows,
                "dim": dim,
                "seed": seed,
                "max_abs": max(rand_diffs) if rand_diffs else 0.0,
            },
            "property": "exp, sigmoid, and exp2 SiLU formulas agree within tolerance in CPU reference math",
            "variants": ["exp", "sigmoid", "exp2"],
        }

    def update_result_json(self, args: dict[str, Any]) -> dict[str, Any]:
        path_arg = args.get("result_path")
        path = Path(path_arg) if path_arg else latest_result_json(self.workspace)
        if path is None:
            raise ToolError("no result.json found")
        if not path.is_absolute():
            path = self.workspace / path
        data = read_json(path) if path.exists() else {}
        metrics = data.setdefault("metrics", {})
        if "best_us" in args:
            best_us = float(args["best_us"])
            metrics["best_us"] = best_us
            metrics["score"] = best_us
        if "status" in args:
            data["status"] = str(args["status"])
        if "verified" in args:
            data["verified"] = bool(args["verified"])
        if "summary" in args:
            data["summary"] = str(args["summary"])
        if "verification" in args and isinstance(args["verification"], dict):
            current = data.setdefault("verification", {})
            current.update(args["verification"])
        if "reflection" in args and isinstance(args["reflection"], dict):
            current = data.setdefault("reflection", {})
            current.update(args["reflection"])
        if "used_harness_artifacts" in args:
            data["used_harness_artifacts"] = list(args["used_harness_artifacts"])
        if "used_harness_tools" in args:
            data["used_harness_tools"] = list(args["used_harness_tools"])
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(path, data)
        return {
            "status": "updated",
            "result_path": str(path),
            "metrics": data.get("metrics", {}),
            "verified": data.get("verified"),
        }


TOOL_DEFS = [
    {
        "name": "inspect_context",
        "description": "Summarize HarnessGym activation, benchmark cases, prior result, baseline, and source hashes.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "run_objective",
        "description": "Run benchmark.py or verifier.py in dev/final mode, locally or through remote_h100.py, and append JSON history.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "workspace_tag": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "remote_health_check",
        "description": "Preflight the configured CUDA host before remote sync: SSH reachability, scratch disk space, and nvidia-smi/GPU visibility.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "string"},
                "key": {"type": "string"},
                "remote_root": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "min_free_mb": {"type": "integer", "minimum": 0},
                "require_remote": {"type": "boolean"},
                "require_gpu": {"type": "boolean"},
                "check_local_cuda": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "sweep_kernel_config",
        "description": "Rollback-safe kernel_config.json sweep with objective runs and best-candidate ranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "keep_best": {"type": "boolean"},
                "configs": {"type": "array"},
                "num_warps": {"type": "array"},
                "num_stages": {"type": "array"},
                "block_size": {"type": "array"},
                "max_configs": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "restore_best_checkpoint",
        "description": "Restore rollback-safe mutable files from the HarnessGym best checkpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "guarded_final_verify",
        "description": "Run an authoritative verifier objective and restore the best checkpoint if the candidate regresses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "max_score": {"type": "number"},
                "restore_on_regression": {"type": "boolean"},
                "restore_files": {"type": "array"},
                "dry_run": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "sweep_launch_overrides",
        "description": "Rollback-safe sweep of per-dimension launch override configs for tunable kernel variants.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "keep_best": {"type": "boolean"},
                "configs": {"type": "array"},
                "dims": {"type": "array"},
                "dry_run": {"type": "boolean"},
                "max_configs": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "sweep_silu_variants",
        "description": "Rollback-safe source sweep of equivalent SiLU formulas, optionally crossed with launch/config overlays.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "keep_best": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
                "variants": {"type": "array"},
                "config_overlays": {"type": "array"},
                "max_candidates": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "probe_silu_approximations",
        "description": "Deterministically check approximate SiLU formulas against RMSNorm+SiLU tolerance on toy, dev, and final-shape proxy inputs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["all", "dev", "final"]},
                "variants": {"type": "array"},
                "sample_rows": {"type": "integer", "minimum": 1},
                "tolerance": {"type": "number"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "sweep_silu_approximations",
        "description": "Rollback-safe source sweep of approximate SiLU rational formulas with deterministic numerical prefiltering before benchmark runs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "keep_best": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
                "variants": {"type": "array"},
                "config_overlays": {"type": "array"},
                "numerical_mode": {"type": "string", "enum": ["all", "dev", "final"]},
                "sample_rows": {"type": "integer", "minimum": 1},
                "tolerance": {"type": "number"},
                "max_candidates": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "joint_source_launch_search",
        "description": "Rollback-safe cross-product search over exact/approximate SiLU source variants and combined per-dimension launch overlays, with optional repeated final confirmation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "keep_best": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
                "source_variants": {"type": "array"},
                "config_overlays": {"type": "array"},
                "max_candidates": {"type": "integer", "minimum": 1},
                "confirm_top_n": {"type": "integer", "minimum": 0},
                "confirm_runs": {"type": "integer", "minimum": 1},
                "confirm_warmup": {"type": "integer", "minimum": 0},
                "confirm_repeats": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "repeat_objective",
        "description": "Run the same dev/final objective multiple times and summarize min/median/max score and per-case timing spread.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "verifier": {"type": "boolean"},
                "warmup": {"type": "integer", "minimum": 0},
                "repeats": {"type": "integer", "minimum": 1},
                "runs": {"type": "integer", "minimum": 1},
                "max_runs": {"type": "integer", "minimum": 1},
                "remote": {"enum": ["auto", "true", "false", True, False]},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "max_score": {"type": "number"},
                "restore_on_regression": {"type": "boolean"},
                "dry_run": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "recommend_next_experiments",
        "description": "Summarize history, held-out shape risks, per-case bests, and next MCP commands for robust optimization.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "top_n": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "rank_history",
        "description": "Rank recorded benchmark/verifier runs by best_us.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "history_path": {"type": "string"},
                "mode": {"type": "string", "enum": ["dev", "final"]},
                "top_n": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "diagnose_source",
        "description": "Inspect kernel.py for known implementation patterns and return source diffs.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "numerical_probe",
        "description": "Run deterministic CPU-side toy and fixed-seed RMSNorm+SiLU formula tolerance checks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rows": {"type": "integer", "minimum": 1},
                "dim": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer"},
                "tolerance": {"type": "number"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "update_result_json",
        "description": "Update latest or specified HarnessGym result.json with metrics, verification, and reflection fields.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "result_path": {"type": "string"},
                "status": {"type": "string"},
                "verified": {"type": "boolean"},
                "best_us": {"type": "number"},
                "summary": {"type": "string"},
                "verification": {"type": "object"},
                "reflection": {"type": "object"},
                "used_harness_artifacts": {"type": "array"},
                "used_harness_tools": {"type": "array"},
            },
            "additionalProperties": False,
        },
    },
]


def read_frame(stdin: Any) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stdin.readline()
        if line == b"":
            return None
        if line in {b"\r\n", b"\n"}:
            break
        key, sep, value = line.decode("ascii", errors="replace").partition(":")
        if sep:
            headers[key.strip().lower()] = value.strip()
    if "content-length" not in headers:
        raise ValueError("missing Content-Length header")
    length = int(headers["content-length"])
    body = stdin.read(length)
    if len(body) != length:
        raise ValueError("short body")
    return json.loads(body.decode("utf-8"))


def write_frame(stdout: Any, payload: dict[str, Any]) -> None:
    body = compact_json(payload).encode("utf-8")
    stdout.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stdout.write(body)
    stdout.flush()


def call_tool(tools: HarnessTools, name: str, args: dict[str, Any]) -> dict[str, Any]:
    if not hasattr(tools, name):
        raise ToolError(f"unknown tool: {name}")
    method = getattr(tools, name)
    return method(args)


def handle_request(tools: HarnessTools, request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    if request_id is None:
        if method == "notifications/initialized":
            return None
        return None
    try:
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        elif method == "tools/list":
            result = {"tools": TOOL_DEFS}
        elif method == "tools/call":
            params = request.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ToolError("tool arguments must be an object")
            try:
                payload = call_tool(tools, str(name), arguments)
                result = {"content": [{"type": "text", "text": pretty_json(payload)}]}
            except Exception as exc:  # noqa: BLE001
                error_payload = {
                    "status": "error",
                    "error": str(exc),
                    "tool": name,
                    "traceback": traceback.format_exc(limit=5),
                }
                result = {"isError": True, "content": [{"type": "text", "text": pretty_json(error_payload)}]}
        else:
            raise ToolError(f"unsupported method: {method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }


def serve(workspace: Path) -> None:
    tools = HarnessTools(workspace)
    while True:
        request = read_frame(sys.stdin.buffer)
        if request is None:
            break
        response = handle_request(tools, request)
        if response is not None:
            write_frame(sys.stdout.buffer, response)


def make_fixture_workspace() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="h100-rmsnorm-mcp-"))
    (tmp / ".harnessgym/runs/run/iterations/1").mkdir(parents=True)
    (tmp / ".harnessgym/runs/run/baseline").mkdir(parents=True)
    write_json(tmp / ".harnessgym/activation.json", {"mcp_servers": [], "skills": []})
    write_json(tmp / ".harnessgym/runs/run/iterations/1/result.json", {"status": "running", "metrics": {}})
    (tmp / ".harnessgym/runs/run/baseline/baseline.stdout.txt").write_text(
        json.dumps({"status": "passed", "best_us": 150.0, "cases": []}) + "\n",
        encoding="utf-8",
    )
    (tmp / "benchmark.py").write_text(
        "CASES = {\n"
        "  'dev': [{'name': 'dev_r2_d4', 'rows': 2, 'dim': 4, 'seed': 1}],\n"
        "  'final': [{'name': 'final_r2_d8', 'rows': 2, 'dim': 8, 'seed': 2}],\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp / "kernel.py").write_text(
        "import triton.language as tl\n"
        "def f(gate):\n"
        "    return gate * tl.sigmoid(gate)\n",
        encoding="utf-8",
    )
    write_json(tmp / "kernel_config.json", {"num_warps": 1, "num_stages": 4, "block_size": 0})
    (tmp / "verifier.py").write_text("", encoding="utf-8")
    (tmp / "remote_h100.py").write_text("", encoding="utf-8")
    return tmp


def self_test() -> int:
    from subprocess import PIPE, Popen

    fixture = make_fixture_workspace()
    proc = Popen(
        [sys.executable, __file__, "--workspace", str(fixture)],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    def request(payload: dict[str, Any]) -> dict[str, Any]:
        body = compact_json(payload).encode("utf-8")
        proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        proc.stdin.flush()
        response = read_frame(proc.stdout)
        assert response is not None
        return response

    def notify(payload: dict[str, Any]) -> None:
        body = compact_json(payload).encode("utf-8")
        proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        proc.stdin.flush()

    try:
        init = request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init["result"]["serverInfo"]["name"] == SERVER_NAME
        notify({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        tools = request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {tool["name"] for tool in tools["result"]["tools"]}
        assert {
            "inspect_context",
            "remote_health_check",
            "numerical_probe",
            "sweep_kernel_config",
            "sweep_launch_overrides",
            "sweep_silu_variants",
            "probe_silu_approximations",
            "sweep_silu_approximations",
            "joint_source_launch_search",
            "repeat_objective",
            "recommend_next_experiments",
            "guarded_final_verify",
            "rank_history",
            "diagnose_source",
        } <= names
        inspect = request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "inspect_context", "arguments": {}},
        })
        inspect_payload = json.loads(inspect["result"]["content"][0]["text"])
        assert inspect_payload["cases"]["dev"][0]["dim"] == 4
        probe = request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "numerical_probe", "arguments": {"rows": 2, "dim": 4, "seed": 7}},
        })
        probe_payload = json.loads(probe["result"]["content"][0]["text"])
        assert probe_payload["status"] == "passed"
        assert "exp2" in probe_payload["variants"]
        remote = request({
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": {
                "name": "remote_health_check",
                "arguments": {"dry_run": True, "host": "gpu.example", "port": "2222"},
            },
        })
        remote_payload = json.loads(remote["result"]["content"][0]["text"])
        assert remote_payload["status"] == "dry_run"
        assert remote_payload["mode"] == "remote"
        assert remote_payload["host"] == "gpu.example"
        approx_probe = request({
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "probe_silu_approximations",
                "arguments": {"mode": "all", "variants": ["rational_m3n2"], "sample_rows": 2},
            },
        })
        approx_probe_payload = json.loads(approx_probe["result"]["content"][0]["text"])
        assert approx_probe_payload["variants"] == ["rational_m3n2"]
        assert "shape_proxy" in approx_probe_payload["results"][0]
        silu = request({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "sweep_silu_variants",
                "arguments": {
                    "dry_run": True,
                    "variants": ["exp", "sigmoid", "exp2"],
                    "config_overlays": [{"num_warps_8192": 16, "num_stages_8192": 2}],
                },
            },
        })
        silu_payload = json.loads(silu["result"]["content"][0]["text"])
        assert silu_payload["status"] == "dry_run"
        assert silu_payload["candidate_count"] == 3
        assert "exp2" in silu_payload["silu_variants"]
        approx_sweep = request({
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "sweep_silu_approximations",
                "arguments": {"dry_run": True, "variants": ["rational_m3n2"], "sample_rows": 2},
            },
        })
        approx_sweep_payload = json.loads(approx_sweep["result"]["content"][0]["text"])
        assert approx_sweep_payload["status"] == "dry_run"
        assert approx_sweep_payload["candidate_count"] == 1
        assert "rational_m3n2" in approx_sweep_payload["approximation_variants"]
        joint = request({
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": {
                "name": "joint_source_launch_search",
                "arguments": {
                    "dry_run": True,
                    "source_variants": ["current", "sigmoid", "rational_m3n2"],
                    "config_overlays": [{}, {"num_warps_8192": 32, "num_stages_8192": 1}],
                    "max_candidates": 8,
                    "confirm_top_n": 1,
                },
            },
        })
        joint_payload = json.loads(joint["result"]["content"][0]["text"])
        assert joint_payload["status"] == "dry_run"
        assert joint_payload["candidate_count"] == 6
        assert "rational_m3n2" in joint_payload["source_variants"]
        repeated = request({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "repeat_objective", "arguments": {"dry_run": True, "runs": 2}},
        })
        repeated_payload = json.loads(repeated["result"]["content"][0]["text"])
        assert repeated_payload["status"] == "dry_run"
        assert repeated_payload["runs"] == 2
        recommend = request({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "recommend_next_experiments", "arguments": {}},
        })
        recommend_payload = json.loads(recommend["result"]["content"][0]["text"])
        assert recommend_payload["status"] == "ready"
        assert recommend_payload["shape_assumptions"]["final_is_held_out"] is True
        launch = request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "sweep_launch_overrides", "arguments": {"dry_run": True, "dims": [1024]}},
        })
        launch_payload = json.loads(launch["result"]["content"][0]["text"])
        assert launch_payload["status"] == "dry_run"
        assert launch_payload["candidate_count"] >= 1
        guarded = request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "guarded_final_verify", "arguments": {"dry_run": True}},
        })
        guarded_payload = json.loads(guarded["result"]["content"][0]["text"])
        assert guarded_payload["status"] == "dry_run"
        bad = request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "run_objective", "arguments": {"mode": "bad"}},
        })
        assert bad["result"].get("isError") is True
        before = (fixture / "kernel_config.json").read_text(encoding="utf-8")
        sweep = request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "sweep_kernel_config",
                "arguments": {
                    "mode": "bad",
                    "configs": [{"num_warps": 1, "num_stages": 4, "block_size": 0}],
                },
            },
        })
        assert sweep["result"].get("isError") is True
        after = (fixture / "kernel_config.json").read_text(encoding="utf-8")
        assert before == after
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        shutil.rmtree(fixture, ignore_errors=True)
    print("self-test passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.self_test:
        return self_test()
    serve(Path(args.workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
