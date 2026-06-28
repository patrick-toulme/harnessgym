#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    result = subprocess.run(
        [sys.executable, "benchmark.py", "--json", "--mode", "final"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    payload = json.loads(result.stdout)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
