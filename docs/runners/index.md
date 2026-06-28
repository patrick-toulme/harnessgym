# Runners

A **runner** is the backend that actually drives an agent through a phase. The
orchestrator doesn't know or care which agent ran the work — it hands the
runner a rendered prompt and a session id, and gets back a `RunnerResult` with
captured stdout, stderr, transcript, return code, timing, and the parsed
session id (`models.RunnerResult`).

Every runner implements the same three-phase contract (`runners.base.Runner`):

| Method | Phase | Session |
| --- | --- | --- |
| `start_attempt` | attempt | opens a **new** session |
| `reflect` | reflection | **resumes** the attempt session |
| `build_tooling` | build / repair | **resumes** the same session |
| `close` | — | tears down any backend process |

Reflection and build resume the attempt session so the agent reflects on work
it genuinely did, not a summary handed back to a cold session.

## Picking a runner

Select with `--runner`:

| Runner | Backend | Use it for |
| --- | --- | --- |
| [`exec`](codex.md) | Codex CLI (`codex exec`) | the recommended MVP path |
| [`claude`](claude.md) | Claude Code CLI (`claude -p`) | running the loop on Claude Code |
| [`fake`](fake.md) | none (deterministic) | tests, demos, install checks |
| [`tui-goal`](fake.md#tui-goal) | Codex PTY (`/goal`) | experimental |

## What's the same across runners

Whatever the backend, HarnessGym applies the same machinery:

- **Process-group timeouts.** The runner launches in its own process group, and
  HarnessGym terminates the whole group on timeout so child benchmark/tool
  processes can't keep pipes open past the deadline.
- **MCP activation + telemetry.** Generated MCP servers are wired into the
  runner's native config and launched through a telemetry wrapper so every
  `tools/call` is logged. See [Activation](../concepts/activation.md) and
  [Telemetry](../concepts/telemetry.md).
- **Skill activation.** Generated skills are symlinked into the runner's native
  skills directory (`.agents/skills` for Codex, `.claude/skills` for Claude).
- **Captured artifacts.** Each phase writes `<phase>.prompt.txt`,
  `.stdout.txt`, `.stderr.txt`, and `.transcript.txt` into the iteration
  directory.

The differences are entirely in *how each backend is invoked and how its MCP
stdio is framed* — covered on the per-runner pages.
