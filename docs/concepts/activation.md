# Activation

Activation is how HarnessGym makes a generated artifact **visible to the runner
in its native form** before each fresh attempt. The source of truth stays under
`.harnessgym/`; activation projects it into whatever Codex or Claude Code
expects (`activation.activate_generated_harness`).

It runs in two places: before every real attempt (Phase 5 of the loop), and
inside [qualification](qualification.md) against the clean-room copy.

## Skills

A skill lives at `.harnessgym/skills/<name>/SKILL.md`. Activation symlinks the
skill directory into the runner's native skills location:

| Runner | Activated path |
| --- | --- |
| Codex (`exec`) | `.agents/skills/<name>` |
| Claude Code (`claude`) | `.claude/skills/<name>` |

Both links are created (`codex_activated_path`, `claude_activated_path`) so the
same registry works regardless of runner. Existing links are replaced
atomically.

## MCP servers

An MCP server is registered through a manifest
(`mcp.json` / `server.json` / `harnessgym-mcp.json`) that declares `name`,
`command`, `args`, `cwd`, `enabled_tools`, and timeouts. Activation normalizes
the manifest and wires the server into the runner:

| Runner | How the server is injected |
| --- | --- |
| Codex (`exec`) | written into project-local `.codex/config.toml`, launched via `codex exec -c mcp_servers...` overrides |
| Claude Code (`claude`) | written into a repo-local Claude MCP config, launched with `--strict-mcp-config --mcp-config <file>` and `--allowedTools=mcp__<server>` |

### The activation quality gate

A server is only advertised to the attempt if it passes a gate
(`_mcp_quality_gate`). For each server, activation:

1. **Smoke checks** it — `initialize` + `tools/list`. A failure becomes a
   warning, not a silent broken server.
2. **Runs its `self_test`** when the manifest declares one. A failed or
   timed-out self-test is recorded.
3. **Checks the tool inventory** is non-empty.

`server["active"]` is `true` only when smoke, self-test, and tool inventory all
pass. The run-level gate aggregates `active_mcp_count`, `inactive_mcp_count`,
and `active_tool_count`. Servers that fail are **not** injected into the
attempt.

The active inventory — server names, per-server smoke/self-test status, and the
enabled tool list — is rendered into the attempt prompt so the agent knows
exactly which tools are live and is told to call them early.

## Telemetry wrappers

Generated MCP servers are never launched bare. Activation wraps them so every
`tools/call` is logged (see [Telemetry](telemetry.md)):

- **Codex** servers run through `harnessgym.mcp_telemetry_proxy`, which
  preserves Content-Length framing while logging calls.
- **Claude Code** servers run through `harnessgym.claude_mcp_bridge`, which also
  translates Claude's newline-delimited MCP stdio to the Content-Length framing
  HarnessGym-generated servers use.

Activation also writes `.harnessgym/runtime/mcp_call.py`. When a Codex exec
session can't see native MCP callables, it can call a generated tool with:

```bash
python3 .harnessgym/runtime/mcp_call.py \
  --server <server> --tool <tool> --arguments '<json-object>'
```

This routes through the telemetry layer and counts as concrete tool use.

## Where it's recorded

Activation results are captured at several points:

- `.harnessgym/activation.json` — the latest activation + quality gate.
- each iteration's `activation.json` — the snapshot used for that attempt.
- each iteration's `post_build_activation.json` — activation after the build.

HarnessGym also infers harness usage from attempt stdout/stderr/transcripts, so
even a timed-out run can report which generated tools were used.

[**Qualification →**](qualification.md){ .md-button }
[**Runners →**](../runners/index.md){ .md-button }
