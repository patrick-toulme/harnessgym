from __future__ import annotations

import math
import random

from kernel_ops import normalized_dot


def reference(a: list[float], b: list[float]) -> float:
    left = math.sqrt(sum(x * x for x in a))
    right = math.sqrt(sum(y * y for y in b))
    if left == 0.0 or right == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (left * right)


def main() -> int:
    rng = random.Random(1337)
    cases = [
        ([1.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 1.0]),
        ([0.0, 0.0], [1.0, 2.0]),
    ]
    for _ in range(200):
        size = rng.randint(2, 32)
        a = [rng.uniform(-10.0, 10.0) for _ in range(size)]
        b = [rng.uniform(-10.0, 10.0) for _ in range(size)]
        cases.append((a, b))

    for index, (a, b) in enumerate(cases):
        got = normalized_dot(a, b)
        expected = reference(a, b)
        if not math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-10):
            print(f"case {index} failed: got {got}, expected {expected}")
            return 1
    print("verifier passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
