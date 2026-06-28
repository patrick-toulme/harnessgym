from __future__ import annotations

import argparse
import json
import time

import numpy as np

from kernel import paged_attention_kernel


BLOCK_SIZE = 16


def _softcap(scores: np.ndarray, cap: float) -> np.ndarray:
    if cap <= 0.0:
        return scores
    return np.tanh(scores / cap) * cap


def reference_paged_attention(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    block_tables: np.ndarray,
    context_lens: np.ndarray,
    scale: float,
    alibi_slopes: np.ndarray,
    window_left: int,
    softcap: float,
) -> tuple[np.ndarray, np.ndarray]:
    batch, heads, dim = q.shape
    kv_heads = k_cache.shape[2]
    out = np.empty((batch, heads, dim), dtype=np.float64)
    lse = np.empty((batch, heads), dtype=np.float64)
    for b in range(batch):
        length = int(context_lens[b])
        start = max(0, length - int(window_left)) if window_left > 0 else 0
        positions = np.arange(start, length, dtype=np.int64)
        if positions.size == 0:
            out[b, :, :] = 0.0
            lse[b, :] = -np.inf
            continue
        physical_blocks = block_tables[b, positions // BLOCK_SIZE]
        offsets = positions % BLOCK_SIZE
        relative_positions = positions.astype(np.float64) - float(length - 1)
        for h in range(heads):
            kv_h = h % kv_heads
            keys = k_cache[physical_blocks, offsets, kv_h, :]
            values = v_cache[physical_blocks, offsets, kv_h, :]
            scores = keys @ q[b, h, :] * scale
            scores = scores + alibi_slopes[h] * relative_positions
            scores = _softcap(scores, softcap)
            max_score = float(np.max(scores))
            weights = np.exp(scores - max_score)
            denom = float(np.sum(weights))
            out[b, h, :] = (weights @ values) / denom
            lse[b, h] = max_score + np.log(denom)
    return out, lse


def make_case(
    seed: int,
    batch: int,
    heads: int,
    kv_heads: int,
    dim: int,
    max_context: int,
    max_blocks: int,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 0.7, size=(batch, heads, dim)).astype(np.float64)
    num_blocks = max_blocks * batch + 31
    k_cache = rng.normal(0.0, 0.5, size=(num_blocks, BLOCK_SIZE, kv_heads, dim)).astype(np.float64)
    v_cache = rng.normal(0.0, 0.5, size=(num_blocks, BLOCK_SIZE, kv_heads, dim)).astype(np.float64)
    context_lens = rng.integers(max(1, max_context // 2), max_context + 1, size=(batch,), dtype=np.int64)
    table_width = int(np.ceil(max_context / BLOCK_SIZE))
    block_tables = np.full((batch, table_width), -1, dtype=np.int64)
    for b in range(batch):
        needed = int(np.ceil(int(context_lens[b]) / BLOCK_SIZE))
        block_tables[b, :needed] = rng.choice(num_blocks, size=needed, replace=False)
    scale = 1.0 / np.sqrt(dim)
    alibi_slopes = rng.uniform(-0.025, 0.025, size=(heads,)).astype(np.float64)
    return q, k_cache, v_cache, block_tables, context_lens, scale, alibi_slopes


def compare_case(seed: int, shape: tuple[int, int, int, int, int, int], window_left: int, softcap: float) -> dict:
    args = make_case(seed, *shape)
    expected_out, expected_lse = reference_paged_attention(*args, window_left, softcap)
    got_out, got_lse = paged_attention_kernel(*args, window_left, softcap)
    out_err = float(np.max(np.abs(got_out - expected_out)))
    lse_err = float(np.max(np.abs(got_lse - expected_lse)))
    ok = (
        got_out.shape == expected_out.shape
        and got_lse.shape == expected_lse.shape
        and np.allclose(got_out, expected_out, rtol=2e-10, atol=2e-10)
        and np.allclose(got_lse, expected_lse, rtol=2e-10, atol=2e-10)
    )
    return {"ok": bool(ok), "out_err": out_err, "lse_err": lse_err, "seed": seed}


def verify() -> list[dict]:
    cases = [
        (101, (1, 2, 1, 8, 19, 3), 0, 0.0),
        (202, (2, 5, 1, 16, 80, 7), 31, 0.0),
        (303, (3, 8, 2, 24, 160, 13), 64, 12.0),
        (404, (4, 12, 4, 32, 257, 21), 127, 8.0),
        (505, (5, 16, 4, 40, 385, 31), 192, 11.0),
    ]
    results = [compare_case(seed, shape, window_left, softcap) for seed, shape, window_left, softcap in cases]
    failures = [result for result in results if not result["ok"]]
    if failures:
        raise AssertionError(f"accuracy failures: {failures[:2]}")
    return results


def benchmark(runs: int = 5) -> dict:
    args = make_case(909, batch=10, heads=16, kv_heads=4, dim=64, max_context=1024, max_blocks=73)
    window_left = 384
    softcap = 10.0
    expected_out, expected_lse = reference_paged_attention(*args, window_left, softcap)
    got_out, got_lse = paged_attention_kernel(*args, window_left, softcap)
    if not np.allclose(got_out, expected_out, rtol=2e-10, atol=2e-10):
        raise AssertionError("benchmark output accuracy failed before timing")
    if not np.allclose(got_lse, expected_lse, rtol=2e-10, atol=2e-10):
        raise AssertionError("benchmark lse accuracy failed before timing")
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        paged_attention_kernel(*args, window_left, softcap)
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "best_ms": min(timings),
        "median_ms": float(np.median(timings)),
        "runs_ms": timings,
        "target_ms": 6.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    accuracy = verify()
    perf = benchmark()
    result = {"status": "passed", "accuracy": accuracy, **perf}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"verifier passed; best_ms={perf['best_ms']:.3f}; target_ms={perf['target_ms']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
