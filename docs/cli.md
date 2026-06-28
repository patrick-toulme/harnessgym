# CLI Reference

HarnessGym exposes two subcommands:

```bash
harnessgym run      # run the iterative attempt → reflect → build → qualify → replay loop
harnessgym compare  # replay plain vs harnessed attempts and write a comparison report
```

A second entry point, `harnessgym-mcp-call`, is a thin client for invoking a
generated MCP tool directly (the same path as
`.harnessgym/runtime/mcp_call.py`).

Timeouts accept human strings like `45m`, `5m`, `90s`, `2h`, or a raw integer
number of seconds.

---

## `harnessgym run`

Run the loop on a task in a workspace.

### Task & workspace

| Flag | Default | Meaning |
| --- | --- | --- |
| `--task PATH` | — | task markdown/text file (mutually exclusive with `--task-text`) |
| `--task-text TEXT` | — | inline task text |
| `--workspace DIR` | `.` | workspace directory for the task |
| `--iterations N` | `3` | maximum iteration count |
| `--run-id ID` | auto | deterministic run id (otherwise timestamp + random suffix) |

A relative `--task` is resolved against the current directory first, then
against `--workspace`.

### Timeouts

| Flag | Default | Meaning |
| --- | --- | --- |
| `--attempt-timeout` | `45m` | attempt-phase timeout |
| `--attempt-timeouts` | — | comma-separated **per-iteration** budgets, e.g. `10s,2m`; overrides `--attempt-timeout` by iteration |
| `--build-timeout` | `20m` | build-phase timeout |
| `--reflection-timeout` | = build | reflection-phase timeout |

### Runner

| Flag | Default | Meaning |
| --- | --- | --- |
| `--runner` | `exec` | `exec`, `claude`, `tui-goal`, or `fake` |
| `--codex-bin` | `codex` | Codex executable (for `exec` / `tui-goal`) |
| `--claude-bin` | `claude` | Claude Code executable (for `claude`) |
| `--claude-model` | — | Claude model alias or full name |
| `--claude-permission-mode` | `bypassPermissions` | Claude Code permission mode |
| `--claude-max-budget-usd` | — | Claude print-mode per-phase spend cap |
| `--claude-extra-arg` | — | extra Claude CLI arg (repeatable) |

See [Runners](runners/index.md) for details.

### Solving & scoring

| Flag | Default | Meaning |
| --- | --- | --- |
| `--stop-score X` | — | stop early when the metric reaches this score |
| `--score-key KEY` | `score` | metric key read from `result.json` metrics/objective |
| `--higher-is-better` | off | treat larger scores as better (default: lower is better) |
| `--build-after-solve` | off | after a verified solve, still reflect/build a reusable harness |

### Optimization & post-attempt scoring

| Flag | Default | Meaning |
| --- | --- | --- |
| `--optimization-mode` | off | treat post-attempt scores as an open-ended objective; report improvement even without reaching `--stop-score` |
| `--post-attempt-command CMD` | — | JSON-emitting command run after **every** attempt, even timed-out ones, for independent scoring |
| `--post-attempt-score-key KEY` | = `--score-key` | metric key read from the post-attempt command output |
| `--post-attempt-timeout` | `2m` | timeout for the post-attempt command |
| `--restore-best` / `--no-restore-best` | restore on | in optimization mode, restore the best independently-scored workspace after the run |

In optimization mode, `summary.json` records the baseline score, best score,
relative improvement, and per-iteration harness usage — so a real improvement
isn't lost when an attempt times out after mutating the workspace but before
writing its result. See [How It Works](how-it-works.md#independent-post-attempt-scoring).

### Task state

| Flag | Default | Meaning |
| --- | --- | --- |
| `--task-state` | `continue` | `continue` compounds task edits across iterations; `reset` restores task files before each new iteration while preserving `.harnessgym` artifacts |

### Harness depth

| Flag | Default | Meaning |
| --- | --- | --- |
| `--harness-depth` | `deep` | `deep` steers reflection/build toward executable instrumentation and richer multi-tool MCP servers; `standard` builds smaller focused artifacts |

### Qualification

| Flag | Default | Meaning |
| --- | --- | --- |
| `--qualify-artifacts` / `--no-qualify-artifacts` | on | validate generated artifacts in a fresh copied workspace before promotion |
| `--artifact-repair-attempts N` | `1` | same-session repair builds to try after a failed qualification |

See [Qualification](concepts/qualification.md).

### Example

```bash
harnessgym run \
  --task task.md --workspace . \
  --iterations 5 --attempt-timeout 5m \
  --runner exec \
  --optimization-mode \
  --score-key best_cycles --stop-score 1 \
  --post-attempt-command "python3 benchmark.py --json --mode final" \
  --post-attempt-score-key best_cycles
```

The command prints the run id, the run-artifacts directory, the final status,
whether it was solved and verified, and — in optimization mode — whether it
improved and the best score.

---

## `harnessgym compare`

Replay plain vs harnessed attempts across copied workspaces and write
`compare_report.json`. Compare runs are **attempt-only** (no reflect/build).

### Required

| Flag | Meaning |
| --- | --- |
| `--workspace-template DIR` | clean workspace template copied for each trial |
| `--task PATH` / `--task-text TEXT` | the task (one is required) |
| `--output-dir DIR` | where trial workspaces and the report are written |

### Trials & budget

| Flag | Default | Meaning |
| --- | --- | --- |
| `--artifact-source PATH` | — | workspace or `.harnessgym` directory holding the generated artifacts |
| `--trials N` | `1` | number of plain **and** harnessed trials |
| `--iterations N` | `1` | iterations per replay trial |
| `--attempt-timeout` | `5m` | attempt timeout per replay iteration |
| `--attempt-timeouts` | — | comma-separated per-iteration budgets |
| `--task-state` | `reset` | task-state mode inside each trial (default keeps attempts comparable) |
| `--overwrite` | off | replace existing trial directories |

### Scoring

| Flag | Default | Meaning |
| --- | --- | --- |
| `--stop-score X` | — | stop a replay early at this score |
| `--score-key KEY` | `score` | metric key from `result.json` |
| `--higher-is-better` | off | larger is better |
| `--post-command CMD` | — | objective command run in each trial workspace after the attempt |
| `--post-score-key KEY` | `score` | metric key from the post-command output |
| `--post-timeout` | `2m` | timeout for the post-command |

### Validity guards

| Flag | Default | Meaning |
| --- | --- | --- |
| `--require-active-harness` / `--no-require-active-harness` | on | mark harnessed trials invalid unless at least one generated MCP tool activates |
| `--require-harness-tool-use` / `--no-require-harness-tool-use` | off | mark harnessed trials invalid unless at least one generated MCP tool call is **recorded** |

See [Replay & Compare](concepts/compare.md) for what makes a comparison
trustworthy.

### Example

```bash
harnessgym compare \
  --workspace-template examples/cpu_attention_autotune_task \
  --task task.md \
  --artifact-source tmp/<run>/.harnessgym \
  --output-dir tmp/cpu_attention_autotune_compare \
  --trials 1 --iterations 5 --attempt-timeout 2m \
  --runner exec \
  --score-key best_cycles --stop-score 1 \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles
```

The command prints the report path and, per group, the trial count, median
attempt and cumulative-attempt durations, best post score, and valid/invalid
post-score counts.
