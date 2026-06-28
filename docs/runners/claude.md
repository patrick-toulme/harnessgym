# Claude Code runner (`claude`)

The `claude` runner drives the Claude Code CLI in headless print mode
(`runners.claude_runner.ClaudeRunner`).

```bash
harnessgym run \
  --task task.md --workspace . --iterations 3 \
  --runner claude --claude-model sonnet
```

## How phases map to Claude Code

| Phase | Command |
| --- | --- |
| attempt | `claude -p --output-format json <prompt>` |
| reflect | `claude -p --output-format json --resume <session_id> <prompt>` |
| build / repair | `claude -p --output-format json --resume <session_id> <prompt>` |

The runner parses Claude's JSON `session_id`, captures stdout/stderr/
transcripts, and enforces process-group timeouts just like the Codex runner.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--claude-bin` | `claude` | path to the Claude Code executable |
| `--claude-model` | — | model alias or full name, e.g. `sonnet` or `opus` |
| `--claude-permission-mode` | `bypassPermissions` | permission mode for autonomous runs |
| `--claude-max-budget-usd` | — | optional per-phase print-mode spend cap |
| `--claude-extra-arg` | — | repeatable escape hatch for newer Claude CLI flags |

`bypassPermissions` is the default because HarnessGym runs Claude Code
autonomously with no human in the loop.

## MCP activation for Claude Code

Qualified MCP servers are written into a repo-local Claude MCP config at
`.harnessgym/claude_mcp_config.json`, and Claude Code is launched with:

```text
--strict-mcp-config --mcp-config .harnessgym/claude_mcp_config.json \
--allowedTools=mcp__<server>
```

`--allowedTools=mcp__<server>` is Claude's server-level permission token, which
grants the generated MCP's tools.

### The stdio framing bridge

HarnessGym-generated MCP servers use **Content-Length** framing (the same
framing Codex expects), while Claude Code speaks **newline-delimited** MCP JSON
over stdio. To bridge the two, each generated server is wrapped with
**`harnessgym.claude_mcp_bridge`**, which translates between the framings and
logs every `tools/call` to `.harnessgym/mcp_calls.jsonl`. The bridge's response
timeout follows each manifest's MCP tool timeout.

This is why a server generated under the Codex runner activates unchanged under
the Claude runner: the bridge handles the framing difference, and the registry
stays runner-agnostic.

## A real validation

The Claude runner was validated end to end on the tensor-layout task, including
hardening to confirm Claude *actually called* the generated MCP tools rather
than merely activating them. See
[Claude Code Runner](../experiments/claude-code-runner-2026-05-20.md) and
[Claude MCP Telemetry](../experiments/claude-mcp-tool-telemetry-2026-05-23.md).

[**Codex runner →**](codex.md){ .md-button }
[**Telemetry →**](../concepts/telemetry.md){ .md-button }
