#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from benchmark import run_suite


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the H100 Triton RMSNorm gate task.")
    parser.add_argument("--mode", choices=("dev", "final"), default="final")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    result = run_suite(args.mode, args.warmup, args.repeats)
    result["verified"] = result.get("status") == "passed"
    print(json.dumps(result, indent=None if args.json else 2, sort_keys=True))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
