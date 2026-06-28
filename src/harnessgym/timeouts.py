from __future__ import annotations

import re


_DURATION_RE = re.compile(r"(?P<value>\d+)(?P<unit>[smhd]?)")
_UNIT_SECONDS = {
    "": 1,
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
}


def parse_timeout(value: str | int | float) -> int:
    """Parse a timeout such as 90, 45m, 1h30m, or 2m10s into seconds."""
    if isinstance(value, bool):
        raise ValueError("timeout must be a duration, not a boolean")
    if isinstance(value, (int, float)):
        seconds = int(value)
        if seconds <= 0:
            raise ValueError("timeout must be positive")
        return seconds

    raw = str(value).strip().lower()
    if not raw:
        raise ValueError("timeout cannot be empty")
    if raw.isdigit():
        seconds = int(raw)
        if seconds <= 0:
            raise ValueError("timeout must be positive")
        return seconds

    total = 0
    index = 0
    for match in _DURATION_RE.finditer(raw):
        if match.start() != index:
            raise ValueError(f"invalid timeout duration: {value!r}")
        index = match.end()
        amount = int(match.group("value"))
        unit = match.group("unit")
        total += amount * _UNIT_SECONDS[unit]

    if index != len(raw) or total <= 0:
        raise ValueError(f"invalid timeout duration: {value!r}")
    return total
