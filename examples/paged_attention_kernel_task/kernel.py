from __future__ import annotations

import numpy as np


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
    """Paged attention decode kernel placeholder.

    This intentionally incomplete implementation ignores the paged KV cache and
    exists so HarnessGym has a real verifier-driven custom-kernel task.
    """
    out = np.zeros_like(q, dtype=np.float64)
    lse = np.zeros(q.shape[:2], dtype=np.float64)
    return out, lse
