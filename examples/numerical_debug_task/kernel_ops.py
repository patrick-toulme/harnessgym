from __future__ import annotations

import math


def normalized_dot(a: list[float], b: list[float]) -> float:
    """Return cosine similarity for two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) + math.sqrt(sum(y * y for y in b))
    if norm == 0.0:
        return 0.0
    return dot / norm
