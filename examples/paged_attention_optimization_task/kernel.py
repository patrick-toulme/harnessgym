from __future__ import annotations

import numpy as np


BLOCK_SIZE = 16


def paged_attention_kernel(
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
    """Correct but intentionally slow paged-attention decode implementation."""
    batch, heads, dim = q.shape
    kv_heads = k_cache.shape[2]
    out = np.empty((batch, heads, dim), dtype=np.float64)
    lse = np.empty((batch, heads), dtype=np.float64)
    for b in range(batch):
        length = int(context_lens[b])
        start = max(0, length - int(window_left)) if window_left > 0 else 0
        if start >= length:
            out[b, :, :] = 0.0
            lse[b, :] = -np.inf
            continue
        for h in range(heads):
            kv_h = h % kv_heads
            scores = []
            values = []
            for pos in range(start, length):
                block = int(block_tables[b, pos // BLOCK_SIZE])
                offset = pos % BLOCK_SIZE
                key = k_cache[block, offset, kv_h, :]
                value = v_cache[block, offset, kv_h, :]
                score = float(np.dot(key, q[b, h, :]) * scale)
                score += float(alibi_slopes[h]) * float(pos - (length - 1))
                if softcap > 0.0:
                    score = float(np.tanh(score / softcap) * softcap)
                scores.append(score)
                values.append(value)
            score_arr = np.array(scores, dtype=np.float64)
            value_arr = np.stack(values, axis=0)
            max_score = float(np.max(score_arr))
            weights = np.exp(score_arr - max_score)
            denom = float(np.sum(weights))
            out[b, h, :] = (weights[:, None] * value_arr).sum(axis=0) / denom
            lse[b, h] = max_score + np.log(denom)
    return out, lse
