# How It Works

This page walks through what happens when you run:

```bash
harnessgym run --task task.md --workspace . --iterations 3 --runner exec
```

Between that command and the `summary.json` written at the end, HarnessGym
runs an iterative loop. Each iteration drives the agent through **five phases**
and then decides whether to stop. There is no model logic inside HarnessGym
itself — the orchestrator coordinates the runner you bring, captures everything
it produces, and enforces the rules that make a generated harness trustworthy.

## The five phases

```text
        ┌─────────┐  ┌─────────┐  ┌───────┐  ┌──────────┐  ┌────────┐
fresh ─▶│ attempt │─▶│ reflect │─▶│ build │─▶│ qualify  │─▶│ replay │─▶ next
session └─────────┘  └─────────┘  └───────┘  └──────────┘  └────────┘   iter
             │            │           │           │             │
        result.json  selected_   .harnessgym/  fresh-copy   activate +
                     improvement   + registry   self-test   sync registry
```

| Phase | Same session? | What it produces | Source |
| --- | --- | --- | --- |
| **attempt** | new session | `result.json` for the primary task | `orchestrator.py`, `prompts.render_attempt_prompt` |
| **reflect** | resumes attempt | `reflection.selected_improvement` | `prompts.render_reflection_prompt` |
| **build** | resumes attempt | one artifact under `.harnessgym/` | `prompts.render_build_prompt` |
| **qualify** | (HarnessGym) + repair session | pass / quarantine in `registry.json` | `qualification.py`, `prompts.render_repair_prompt` |
| **replay** | next new session | activated skills + MCP config | `activation.py` |

The attempt is a *fresh* session every iteration. Reflection and build
**resume that same session** — they run `codex exec resume <id>` or
`claude -p --resume <id>` so the agent reflects on work it actually did, not a
summary. Qualification runs inside HarnessGym (and may open one repair session).
Replay then opens the next fresh session with the harness already wired in.

---

## Phase 1: Attempt

Each iteration starts by preparing the workspace and opening a new runner
session for the primary task.

1. **Task state** is prepared. With `--task-state continue` (the default for
   `run`) the working tree compounds across iterations. With `reset`,
   HarnessGym snapshots the non-`.harnessgym` files before the run and restores
   them before iterations 2..N — so each fresh session faces the original task
   with only the *generated* harness carried forward. (`task_state.py`)
2. The **registry is loaded and synced from disk** (`sync_registry_from_files`)
   so anything built in a previous iteration is known.
3. The harness is **activated** (Phase 5 of the previous iteration, replayed
   here) and an `activation.json` snapshot is written into the iteration
   directory.
4. An **initial `result.json`** is created, and the attempt prompt is rendered
   and sent to the runner.

The attempt prompt (`render_attempt_prompt`) hands the agent the task text, the
registered-artifact context, the active skill/MCP inventory, and an explicit
instruction: if an active MCP exposes benchmark / verifier / search / ranking /
rollback tools relevant to the task, **call them early** rather than
re-deriving the workflow by hand. It also tells the agent exactly where to
write its machine-readable result:

```json
{
  "status": "solved | blocked | incomplete | tooling_built | failed",
  "verified": true,
  "summary": "...",
  "reflection": { "selected_improvement": { "kind": "...", "name": "...", "reason": "...", "target_path": "..." } },
  "verification": { "status": "passed | failed | not_run", "tooling_tests": [ ... ] },
  "metrics": { "score": 0 }
}
```

HarnessGym launches the runner in a **process group** and terminates the whole
group on timeout, so spawned benchmark or tool children can't keep pipes open
past the deadline.

### Independent post-attempt scoring

For time-boxed optimization, the attempt can be killed mid-edit — after it
mutated the workspace but before it wrote `result.json`. To stop a real
improvement from being lost, `--post-attempt-command` runs an objective
JSON-emitting command after **every** attempt, even a timed-out one
(`postprocess.py`). The score it reports is what HarnessGym trusts for
checkpointing and stop conditions.

```bash
--post-attempt-command "python3 benchmark.py --json --mode final" \
--post-attempt-score-key best_cycles
```

When `--optimization-mode` is on, HarnessGym keeps a **best checkpoint**
(`checkpoints.BestCheckpointManager`): each time an attempt produces a better
independently-scored workspace, that workspace is snapshotted. At the end of
the run the best checkpoint is restored (unless `--no-restore-best`), while
`.harnessgym`, `.codex`, `.agents`, and `.claude` activation state is left
intact.

---

## Phase 2: Reflect

If the task isn't already solved-and-verified, HarnessGym **resumes the same
session** and asks one question (`render_reflection_prompt`): *based on the work
you just did, what single skill, MCP server, verifier, fixture, script, docs, or
tool would most improve solve time or make this task more solvable next time?*

The prompt is opinionated about leverage. For kernel / numerical / compiler /
search / verifier-driven tasks it steers toward an **agent-native harness
package** — a skill plus a stdio MCP server that exposes
verifier/debugging/search commands — over a disconnected note. For follow-on
iterations it asks the agent to **extend and harden the existing suite** rather
than build a disconnected wrapper.

The agent writes its choice into `reflection.selected_improvement` in
`result.json`. HarnessGym reads it back with `extract_selected_improvement`,
falling back to the reflection transcript if the structured field is missing.

---

## Phase 3: Build

Still in the same session, HarnessGym sends the build prompt
(`render_build_prompt`) carrying the selected improvement. The rules here are
what make a generated artifact reusable:

- **Build only the one selected improvement.** Don't keep solving the task.
- **Skills** go to `.harnessgym/skills/<name>/SKILL.md` with frontmatter.
- **MCP servers** go to `.harnessgym/mcp/<name>/` with a manifest
  (`mcp.json`, `server.json`, or `harnessgym-mcp.json`) declaring `name`,
  `command`, `args`, `cwd`, `enabled_tools`, and timeouts.
- **MCP servers must speak Content-Length-framed stdio JSON-RPC**
  (`Content-Length: <bytes>\r\n\r\n<body>`). Newline-delimited JSON is not
  enough for Codex or Claude MCP activation, and the build prompt says so.
- **Every generated tool ships deterministic tests.** For MCP servers, a
  manifest `self_test` entry lets HarnessGym run the test during qualification
  and activation. For numerical/kernel/compiler work the tests must include toy
  cases, tolerance checks, fixed-seed randomized cases, and fast/dev vs
  final/held-out modes when applicable.

With `--harness-depth deep` (the default), the prompt pushes toward
capability-creating instrumentation — a multi-tool MCP server spanning
observation, verification, search, rollback-safe mutation, history comparison,
and next-experiment ranking — rather than a notes file. `--harness-depth
standard` asks for a smaller, focused artifact.

After the build, HarnessGym re-syncs the registry from the files on disk
(`sync_registry_from_files`), so `.harnessgym/registry.json` always reflects
what actually exists.

---

## Phase 4: Qualify (and repair, or quarantine)

A built artifact is not trusted until it survives a **clean-room replay**. This
is the gate that separates "the agent wrote some files" from "a fresh session
can actually use this."

At the very start of the run, HarnessGym captured a clean copy of the task
workspace (`capture_clean_workspace_template`). Now, for each build
(`qualify_generated_harness`):

1. Copy that clean template into
   `.harnessgym/runs/<run_id>/iterations/<n>/qualification/attempt_<k>/`.
2. Copy **only** the reusable `.harnessgym/` bundle into it — nothing else from
   the dirty training workspace.
3. Activate the generated skills and MCP servers there.
4. Run the MCP self-tests and check the quality gate (non-empty tool
   inventory, smoke check, self-test passing).

If qualification **passes**, the artifacts are marked qualified in
`registry.json` and the run continues.

If it **fails**, HarnessGym sends the exact failure report back into the same
session (`render_repair_prompt`) and allows up to `--artifact-repair-attempts`
repair builds (default `1`). The repair prompt is narrow: *fix the generated
artifact so it activates in a fresh copied workspace — don't keep solving the
original task.*

Anything still failing after repairs is **quarantined** in `registry.json`
(`quarantine_artifacts`): the files stay on disk with qualification metadata
and the failure-report path, but the artifact is hidden from future attempt
prompts and is never injected into a runner session. A broken harness can never
quietly pollute the next attempt.

[**More on qualification →**](concepts/qualification.md){ .md-button }

---

## Phase 5: Replay (activation)

Before the *next* fresh attempt, HarnessGym makes the promoted registry visible
to the runner in its native form (`activate_generated_harness`):

- **Skills** from `.harnessgym/skills/<name>/SKILL.md` are symlinked into
  `.agents/skills/<name>` for Codex and `.claude/skills/<name>` for Claude
  Code.
- **MCP manifests** are written into project-local `.codex/config.toml` for
  Codex, or a repo-local Claude MCP config for Claude Code.
- Each MCP server is **smoke-checked** with `initialize` + `tools/list` and
  must pass its `self_test` before it is advertised. Failures become warnings,
  not silent broken servers.
- Generated MCP servers are launched through HarnessGym's telemetry layer (a
  Content-Length proxy for Codex, a stdio framing bridge for Claude) so every
  `tools/call` is logged.

The source of truth stays under `.harnessgym/`; activation just projects it
into whatever the runner expects. The next fresh session sees the skills and
MCP tools as if they had always been there.

[**More on activation →**](concepts/activation.md){ .md-button }

---

## Stopping

After each iteration HarnessGym checks the result and decides whether to
continue:

- **Solved + verified** (`status: "solved"` and `verified: true`, or a passed
  verification object) with no `--stop-score` set → stop. (Use
  `--build-after-solve` to still reflect/build a reusable harness on a solved
  run.)
- **`--stop-score` reached** on the chosen `--score-key` → stop. By default
  lower is better; pass `--higher-is-better` to flip it.
- Otherwise → run the next iteration.

When no stop condition fires (common for open-ended optimization with an
intentionally unreachable `--stop-score 1`), all iterations run and the final
status is typically `incomplete` or `tooling_built`.

---

## What gets written

Every run is fully inspectable on disk:

```text
.harnessgym/
├── registry.json                      # synced inventory of generated artifacts
├── activation.json                    # latest activation + quality gate
├── mcp_calls.jsonl                    # every generated MCP tools/call
├── skills/ mcp/ tools/ verifiers/ …   # the reusable artifacts themselves
└── runs/<run_id>/
    ├── run_config.json                # the exact config used
    ├── summary.json                   # final status, best score, optimization, quality gates
    └── iterations/<n>/
        ├── result.json
        ├── activation.json
        ├── <phase>.prompt.txt / .stdout.txt / .stderr.txt / .transcript.txt
        └── qualification/attempt_<k>/qualification.json
```

`summary.json` records the final status, best score, baseline vs best
optimization delta, per-iteration harness usage, and the harness quality-gate
status — enough to reconstruct what the loop decided and why.

---

## What to read next

- [Getting Started](getting-started.md) — run the offline demo and watch the loop.
- [CLI Reference](cli.md) — every `run` and `compare` flag.
- [Artifacts & Registry](concepts/artifacts.md) — where generated tools live and how they're tracked.
- [Philosophy](philosophy.md) — the "why" behind the five phases.
