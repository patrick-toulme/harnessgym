from __future__ import annotations

import numpy as np


def selective_scan_kernel(
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Fused selective-scan kernel placeholder.

    The current implementation is intentionally incomplete for the HarnessGym
    demo. It has the right shape behavior but ignores the recurrent state.
    """
    return D.reshape(1, 1, -1) * u
