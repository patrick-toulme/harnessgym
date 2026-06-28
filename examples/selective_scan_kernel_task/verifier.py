from __future__ import annotations

import time

import numpy as np

from kernel import selective_scan_kernel


def reference_selective_scan(
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    batch, seq, channels = u.shape
    out = np.empty_like(u, dtype=np.float64)
    state = np.zeros((batch, channels), dtype=np.float64)
    for t in range(seq):
        decay = np.exp(delta[:, t, :] * A.reshape(1, channels))
        state = decay * state + B.reshape(1, channels) * u[:, t, :]
        out[:, t, :] = C.reshape(1, channels) * state + D.reshape(1, channels) * u[:, t, :]
    return out


def make_case(seed: int, batch: int, seq: int, channels: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, size=(batch, seq, channels)).astype(np.float64)
    delta = rng.uniform(0.0, 0.08, size=(batch, seq, channels)).astype(np.float64)
    A = -rng.uniform(0.1, 1.0, size=(channels,)).astype(np.float64)
    B = rng.normal(0.0, 0.2, size=(channels,)).astype(np.float64)
    C = rng.normal(0.0, 0.2, size=(channels,)).astype(np.float64)
    D = rng.normal(0.0, 0.1, size=(channels,)).astype(np.float64)
    return u, delta, A, B, C, D


def check_accuracy() -> None:
    for seed, shape in enumerate([(1, 4, 3), (2, 9, 7), (3, 17, 11), (4, 31, 19)], start=10):
        args = make_case(seed, *shape)
        expected = reference_selective_scan(*args)
        got = selective_scan_kernel(*args)
        if got.shape != expected.shape:
            raise AssertionError(f"shape mismatch: got {got.shape}, expected {expected.shape}")
        max_abs = float(np.max(np.abs(got - expected)))
        if not np.allclose(got, expected, rtol=2e-10, atol=2e-10):
            raise AssertionError(f"accuracy mismatch for shape {shape}: max_abs={max_abs:.3e}")


def check_performance() -> None:
    args = make_case(2026, batch=8, seq=384, channels=256)
    selective_scan_kernel(*args)
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        got = selective_scan_kernel(*args)
        runs.append(time.perf_counter() - start)
    expected = reference_selective_scan(*[arg[:, :12, :] if arg.ndim == 3 else arg for arg in args])
    small_got = selective_scan_kernel(*[arg[:, :12, :] if arg.ndim == 3 else arg for arg in args])
    if not np.allclose(small_got, expected, rtol=2e-10, atol=2e-10):
        raise AssertionError("large-case prefix accuracy failed")
    best = min(runs)
    if best > 0.08:
        raise AssertionError(f"kernel too slow: best={best:.4f}s, threshold=0.0800s")
    print(f"performance best={best:.4f}s threshold=0.0800s")


def main() -> int:
    check_accuracy()
    check_performance()
    print("verifier passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
