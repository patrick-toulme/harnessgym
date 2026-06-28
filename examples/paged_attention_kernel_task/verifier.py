from __future__ import annotations

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
        logical_blocks = positions // BLOCK_SIZE
        offsets = positions % BLOCK_SIZE
        physical_blocks = block_tables[b, logical_blocks]
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
    num_blocks = max_blocks * batch + 17
    k_cache = rng.normal(0.0, 0.5, size=(num_blocks, BLOCK_SIZE, kv_heads, dim)).astype(np.float64)
    v_cache = rng.normal(0.0, 0.5, size=(num_blocks, BLOCK_SIZE, kv_heads, dim)).astype(np.float64)
    context_lens = rng.integers(max(1, max_context // 3), max_context + 1, size=(batch,), dtype=np.int64)
    table_width = int(np.ceil(max_context / BLOCK_SIZE))
    block_tables = np.full((batch, table_width), -1, dtype=np.int64)
    for b in range(batch):
        needed = int(np.ceil(int(context_lens[b]) / BLOCK_SIZE))
        block_tables[b, :needed] = rng.choice(num_blocks, size=needed, replace=False)
    scale = 1.0 / np.sqrt(dim)
    alibi_slopes = rng.uniform(-0.025, 0.025, size=(heads,)).astype(np.float64)
    return q, k_cache, v_cache, block_tables, context_lens, scale, alibi_slopes


def compare_case(seed: int, shape: tuple[int, int, int, int, int, int], window_left: int, softcap: float) -> None:
    args = make_case(seed, *shape)
    expected_out, expected_lse = reference_paged_attention(*args, window_left, softcap)
    got_out, got_lse = paged_attention_kernel(*args, window_left, softcap)
    if got_out.shape != expected_out.shape:
        raise AssertionError(f"output shape mismatch: got {got_out.shape}, expected {expected_out.shape}")
    if got_lse.shape != expected_lse.shape:
        raise AssertionError(f"lse shape mismatch: got {got_lse.shape}, expected {expected_lse.shape}")
    out_err = float(np.max(np.abs(got_out - expected_out)))
    lse_err = float(np.max(np.abs(got_lse - expected_lse)))
    if not np.allclose(got_out, expected_out, rtol=2e-10, atol=2e-10):
        raise AssertionError(f"output mismatch seed={seed}: max_abs={out_err:.3e}")
    if not np.allclose(got_lse, expected_lse, rtol=2e-10, atol=2e-10):
        raise AssertionError(f"lse mismatch seed={seed}: max_abs={lse_err:.3e}")


def check_accuracy() -> None:
    compare_case(101, (1, 2, 1, 8, 19, 3), window_left=0, softcap=0.0)
    compare_case(202, (2, 5, 1, 16, 80, 7), window_left=31, softcap=0.0)
    compare_case(303, (3, 8, 2, 24, 160, 13), window_left=64, softcap=12.0)
    compare_case(404, (4, 12, 4, 32, 257, 21), window_left=127, softcap=8.0)


def check_performance() -> None:
    args = make_case(909, batch=8, heads=12, kv_heads=4, dim=48, max_context=768, max_blocks=53)
    window_left = 256
    softcap = 10.0
    expected_out, expected_lse = reference_paged_attention(*args, window_left, softcap)
    got_out, got_lse = paged_attention_kernel(*args, window_left, softcap)
    if not np.allclose(got_out, expected_out, rtol=2e-10, atol=2e-10):
        raise AssertionError("benchmark output accuracy failed before timing")
    if not np.allclose(got_lse, expected_lse, rtol=2e-10, atol=2e-10):
        raise AssertionError("benchmark lse accuracy failed before timing")
    runs = []
    for _ in range(4):
        start = time.perf_counter()
        paged_attention_kernel(*args, window_left, softcap)
        runs.append(time.perf_counter() - start)
    best = min(runs)
    if best > 0.200:
        raise AssertionError(f"kernel too slow: best={best:.4f}s, threshold=0.2000s")
    print(f"performance best={best:.4f}s threshold=0.2000s")


def main() -> int:
    check_accuracy()
    check_performance()
    print("verifier passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
