#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CLAUDE_BIN="${CLAUDE_BIN:-claude}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ATTEMPT_TIMEOUT="${ATTEMPT_TIMEOUT:-5m}"
POST_TIMEOUT="${POST_TIMEOUT:-2m}"
TRIALS="${TRIALS:-1}"
ITERATIONS="${ITERATIONS:-1}"
REQUIRE_HARNESS_TOOL_USE="${REQUIRE_HARNESS_TOOL_USE:-1}"
WAIT_FOR_CLAUDE="${WAIT_FOR_CLAUDE:-0}"
PREFLIGHT_RETRY_SECONDS="${PREFLIGHT_RETRY_SECONDS:-300}"
PREFLIGHT_MAX_WAIT_SECONDS="${PREFLIGHT_MAX_WAIT_SECONDS:-14400}"
RUN_STAMP="$("$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
PY
)"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/tensor_layout_claude_compare_${RUN_STAMP}}"

PREFLIGHT_OUT="$(mktemp "${TMPDIR:-/tmp}/harnessgym-claude-preflight.XXXXXX")"
trap 'rm -f "$PREFLIGHT_OUT"' EXIT

print_preflight_failure() {
  echo "Claude Code preflight failed; not running compare." >&2
  "$PYTHON_BIN" - "$PREFLIGHT_OUT" >&2 <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").strip()
try:
    payload = json.loads(text)
except json.JSONDecodeError:
    print(text)
else:
    status = payload.get("api_error_status")
    result = payload.get("result")
    if status is not None:
        print(f"api_error_status: {status}")
    if result:
        print(result)
PY
}

echo "Checking Claude Code availability with ${CLAUDE_BIN}..."
preflight_started="$(date +%s)"
while true; do
  if "$CLAUDE_BIN" -p --output-format json "Reply OK." >"$PREFLIGHT_OUT" 2>&1; then
    break
  fi
  if [[ "$WAIT_FOR_CLAUDE" != "1" ]]; then
    print_preflight_failure
    exit 1
  fi
  now="$(date +%s)"
  elapsed="$((now - preflight_started))"
  if (( elapsed >= PREFLIGHT_MAX_WAIT_SECONDS )); then
    echo "Claude Code preflight still failing after ${elapsed}s." >&2
    print_preflight_failure
    exit 1
  fi
  print_preflight_failure
  echo "Retrying Claude Code preflight in ${PREFLIGHT_RETRY_SECONDS}s..." >&2
  sleep "$PREFLIGHT_RETRY_SECONDS"
done
if [[ "$WAIT_FOR_CLAUDE" == "1" ]]; then
  echo "Claude Code preflight passed after $(( $(date +%s) - preflight_started ))s."
fi

echo "Running HarnessGym tensor-layout Claude compare..."
TOOL_USE_ARGS=()
CHECKER_TOOL_USE_ARGS=()
if [[ "$REQUIRE_HARNESS_TOOL_USE" == "1" ]]; then
  TOOL_USE_ARGS+=(--require-harness-tool-use)
  CHECKER_TOOL_USE_ARGS+=(--min-mcp-calls 1)
fi
PYTHONPATH=src "$PYTHON_BIN" -m harnessgym.cli compare \
  --workspace-template examples/tensor_layout_pipeline_task \
  --task task.md \
  --artifact-source examples/tensor_layout_harness_artifacts/.harnessgym \
  --output-dir "$OUTPUT_DIR" \
  --trials "$TRIALS" \
  --iterations "$ITERATIONS" \
  --attempt-timeout "$ATTEMPT_TIMEOUT" \
  --runner claude \
  --claude-bin "$CLAUDE_BIN" \
  --score-key best_cycles \
  --stop-score 1 \
  --task-state continue \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles \
  --post-timeout "$POST_TIMEOUT" \
  "${TOOL_USE_ARGS[@]}" \
  --overwrite

echo "Compare report: ${OUTPUT_DIR}/compare_report.json"
"$PYTHON_BIN" examples/tensor_layout_harness_artifacts/check_claude_compare_report.py \
  "${OUTPUT_DIR}/compare_report.json" \
  --min-active-mcp 1 \
  --min-active-tools 15 \
  "${CHECKER_TOOL_USE_ARGS[@]}"
