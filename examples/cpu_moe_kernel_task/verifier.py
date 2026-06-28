#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

import benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(benchmark.CASES), default="final")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = benchmark.load_config()
    result = benchmark.evaluate_config(config, args.mode)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{args.mode} status={result['status']} "
            f"best_cycles={result['best_cycles']} max_abs={result['max_abs']}"
        )
        for case in result.get("cases", []):
            print(f"  {case['name']}: {case['status']} cycles={case['best_cycles']} max_abs={case['max_abs']}")
        for error in result.get("errors", []):
            print(f"  error: {error}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
