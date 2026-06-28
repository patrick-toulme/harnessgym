# MCP Telemetry

Activation alone proves nothing — a tool can be live and never touched. So
HarnessGym records **every generated MCP `tools/call`** as it happens, and
surfaces that record where it matters: in the attempt result, in
`summary.json`, and in compare reports.

## The call log

Every generated MCP `tools/call` is appended as one compact JSON line to:

```text
.harnessgym/mcp_calls.jsonl
```

Each event carries the server name, tool name, an argument-key summary, the
duration, the status, and the result size. The log is written by whichever
telemetry wrapper launched the server:

- **Codex** — `harnessgym.mcp_telemetry_proxy`, a Content-Length-preserving
  JSON-RPC proxy in front of the generated server.
- **Claude Code** — `harnessgym.claude_mcp_bridge`, which both logs calls and
  translates Claude's newline-delimited MCP stdio into the Content-Length
  framing the generated servers speak.

Because the wrappers sit on the JSON-RPC path, the log captures real calls, not
the agent *claiming* it made calls. The exception is a manual bypass: if a
session writes its own ad-hoc JSON-RPC client or launches the server file
directly, those calls skip the wrapper and aren't logged — which is exactly why
the prompts forbid it and tell the agent to use the
`.harnessgym/runtime/mcp_call.py` helper instead.

## Summaries

`mcp_telemetry.summarize_mcp_call_events` rolls the log into a structured
summary:

```json
{
  "path": ".harnessgym/mcp_calls.jsonl",
  "count": 14,
  "successful_count": 13,
  "failed_count": 1,
  "status_counts": { "completed": 13, "error": 1 },
  "called_tools": ["autotune", "evaluate_final", "rank_candidates", "..."],
  "servers": ["cpu_attention_autotune"],
  "samples": [ { "server": "...", "tool_name": "...", "status": "completed", ... } ]
}
```

This is what flows into `compare_report.json` as `mcp_call_count`,
`mcp_called_tools`, and the per-group `mcp_telemetry`, and into the run
`summary.json` harness-usage section.

## Why it matters for evidence

The telemetry is what lets a comparison demand real usage:

- `--require-harness-tool-use` (on `compare`) makes a harnessed trial count
  **only** if at least one generated tool call is recorded.
- The H100 experiments use this to confirm, for example, **10 generated MCP
  calls in the next fresh Codex session** — not just an activated server.

Without the log, "the harness helped" is an assertion. With it, it's a record
you can read back line by line.

## Reading it yourself

```python
from pathlib import Path
from harnessgym.mcp_telemetry import read_mcp_call_events, summarize_mcp_call_events

events = read_mcp_call_events(Path("."))           # raw events
summary = summarize_mcp_call_events(Path("."))     # rolled-up summary
print(summary["count"], summary["called_tools"])
```

[**Replay & Compare →**](compare.md){ .md-button }
[**Activation →**](activation.md){ .md-button }
