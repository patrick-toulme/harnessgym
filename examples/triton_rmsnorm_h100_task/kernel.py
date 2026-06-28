from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "kernel_config.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "num_warps": 8,
    "num_stages": 3,
    "block_size": 0,
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {**DEFAULT_CONFIG, **data}


def validate_config(config: dict[str, Any], dim: int) -> list[str]:
    errors: list[str] = []
    if int(config.get("num_warps", 0)) not in {1, 2, 4, 8, 16, 32}:
        errors.append("num_warps must be one of 1, 2, 4, 8, 16, 32")
    if int(config.get("num_stages", 0)) not in {1, 2, 3, 4, 5}:
        errors.append("num_stages must be one of 1, 2, 3, 4, 5")
    block_size = int(config.get("block_size", 0))
    if block_size < 0:
        errors.append("block_size must be 0 or a positive power of two")
    if block_size:
        if block_size & (block_size - 1):
            errors.append("block_size must be a power of two")
        if block_size < dim:
            errors.append("block_size must be >= dim")
        if block_size > 16384:
            errors.append("block_size must be <= 16384")
    return errors


@triton.jit
def _rmsnorm_silu_gate_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < dim
    base = row * dim + offsets

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + base, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    rms = tl.rsqrt(tl.sum(x * x, axis=0) / dim + eps)
    silu_gate = gate / (1.0 + tl.exp(-gate))
    y = x * rms * weight * silu_gate

    tl.store(out_ptr + base, y, mask=mask)


def fused_rmsnorm_silu_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-5,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor")
    if gate.shape != x.shape:
        raise ValueError("gate must have the same shape as x")
    if weight.ndim != 1 or weight.numel() != x.shape[1]:
        raise ValueError("weight must be a 1D tensor matching x.shape[1]")
    if not x.is_cuda or not gate.is_cuda or not weight.is_cuda:
        raise ValueError("x, gate, and weight must be CUDA tensors")
    if not x.is_contiguous() or not gate.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x, gate, and weight must be contiguous")

    config = {**DEFAULT_CONFIG, **(config or load_config())}
    rows, dim = x.shape
    errors = validate_config(config, dim)
    if errors:
        raise ValueError("; ".join(errors))

    configured_block = int(config.get("block_size", 0))
    block_size = configured_block or triton.next_power_of_2(dim)
    out = torch.empty_like(x)
    _rmsnorm_silu_gate_kernel[(rows,)](
        x,
        gate,
        weight,
        out,
        rows,
        dim,
        float(eps),
        BLOCK_SIZE=block_size,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
    return out


def torch_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    x32 = x.float()
    gate32 = gate.float()
    weight32 = weight.float()
    rms = torch.rsqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + eps)
    out = x32 * rms * weight32 * torch.nn.functional.silu(gate32)
    return out.to(dtype=x.dtype)
