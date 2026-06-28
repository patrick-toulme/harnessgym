# Codex runner (`exec`)

The `exec` runner is the recommended MVP backend. It drives the OpenAI Codex
CLI (`runners.exec_runner.ExecRunner`).

```bash
harnessgym run --task task.md --workspace . --iterations 3 --runner exec
```

## How phases map to Codex

| Phase | Command |
| --- | --- |
| attempt | `codex exec <prompt>` |
| reflect | `codex exec resume <session_id> <prompt>` |
| build / repair | `codex exec resume <session_id> <prompt>` |

The attempt opens a fresh Codex session; reflection and build resume it via the
session id Codex reports. It uses ordinary autonomous prompts — it does **not**
assume `codex exec "/goal ..."` sets a goal.

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--codex-bin` | `codex` | path to the Codex executable |

## MCP activation for Codex

Generated MCP servers are injected as `codex exec -c mcp_servers...` overrides,
because `codex exec` loads the user config by default and HarnessGym should not
have to mutate global Codex config for a repo-local generated server.

Each generated server is launched through **`harnessgym.mcp_telemetry_proxy`**,
which preserves Content-Length MCP framing while logging each `tools/call` to
`.harnessgym/mcp_calls.jsonl`.

### The `mcp_call.py` helper

Activation also writes `.harnessgym/runtime/mcp_call.py`. When a `codex exec`
worker can't see native MCP callables in its session, it should call generated
tools through this helper rather than writing an ad-hoc JSON-RPC client:

```bash
python3 .harnessgym/runtime/mcp_call.py \
  --server <server> --tool <tool> --arguments '<json-object>'
```

This routes through the telemetry proxy, so the call is logged and counts as
concrete harness tool use. Bypassing it — launching `.harnessgym/mcp/...`
server files directly — loses telemetry and does not count as verified tool
use.

## Timeout handling

HarnessGym launches Codex in a process group and terminates the whole group on
timeout, so spawned benchmark/tool children cannot keep pipes open
indefinitely. A timed-out attempt still has its workspace scored when
`--post-attempt-command` is set — see [How It Works](../how-it-works.md).

[**Claude Code runner →**](claude.md){ .md-button }
[**Activation →**](../concepts/activation.md){ .md-button }
