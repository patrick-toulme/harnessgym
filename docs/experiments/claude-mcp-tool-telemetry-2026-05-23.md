# Claude MCP Tool-Use Telemetry - 2026-05-23

This note records the follow-up hardening after the first Claude Code
tensor-layout validation showed activated MCP tools but no machine-readable
evidence that Claude actually called them.

## Implementation

- `harnessgym.claude_mcp_bridge` now logs every generated MCP
  `tools/call` to `.harnessgym/mcp_calls.jsonl`.
- Each event records server name, tool name, argument-key summary, duration,
  status, error message, and result size.
- The Claude runner passes the active MCP server name and manifest tool timeout
  into the bridge.
- The Codex `exec` runner now launches active generated MCP servers through
  `harnessgym.mcp_telemetry_proxy`, so Codex worker runs record the same
  JSONL tool-call telemetry without changing the generated MCP server.
- `harnessgym compare` reads the JSONL telemetry and writes
  `mcp_telemetry`, `mcp_call_count`, and `mcp_called_tools` into
  `compare_report.json`.
- `harnessgym compare --require-harness-tool-use` marks harnessed trials
  invalid when generated MCP tools activate but no tool call is recorded.
- Attempt prompts now explicitly tell the agent that MCP activation alone does
  not count as harness use.

## Validation

Unit suite:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
63 passed, 6 subtests passed in 8.07s
```

Syntax checks:

```bash
bash -n examples/tensor_layout_harness_artifacts/run_claude_compare.sh
PYTHONPATH=src .venv/bin/python examples/tensor_layout_harness_artifacts/check_claude_compare_report.py --help >/dev/null
```

## Real Claude Attempt

Started a stricter real replay:

```bash
ATTEMPT_TIMEOUT=10m TRIALS=2 ITERATIONS=1 POST_TIMEOUT=2m \
REQUIRE_HARNESS_TOOL_USE=1 \
OUTPUT_DIR=tmp/tensor_layout_claude_tooluse_10m_2trial_20260523T184552Z \
examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```

The first two plain trials were valid:

- Trial 1: `post_score=1,495,982`, attempt duration `397.284s`.
- Trial 2: `post_score=1,495,982`, attempt duration `600.222s`.

Harnessed trial 1 immediately proved the new telemetry path by recording real
generated MCP tool calls:

```text
resume_search_history
apply_best_verified
trace_summary
rank_next_experiments
local_neighborhood_search
bounded_exhaustive_search
search_plans
```

The partial run was stopped because telemetry exposed a HarnessGym bug: the
Claude bridge used a fixed 30 second MCP response timeout even though the
generated tensor-plan manifest declares longer tool timeouts. Several
`bounded_exhaustive_search` calls were clipped by the bridge. The bridge now
honors each active server's manifest tool timeout.

## Rerun Status

After the timeout fix, a fresh 2-trial rerun was started:

```bash
ATTEMPT_TIMEOUT=10m TRIALS=2 ITERATIONS=1 POST_TIMEOUT=2m \
REQUIRE_HARNESS_TOOL_USE=1 \
OUTPUT_DIR=tmp/tensor_layout_claude_tooluse_timeoutfix_10m_2trial_20260523T190630Z \
examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```

Claude Code returned:

```text
You're out of extra usage · resets 7:40pm (America/New_York)
```

The resulting report correctly marked all trials invalid because the runner
failed before useful attempt work. That rerun is not a performance result. The
next valid replay should use the same command after the Claude quota reset.
