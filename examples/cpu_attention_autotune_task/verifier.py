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
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")
        return result.returncode
    payload = json.loads(result.stdout)
    return 0 if payload.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
