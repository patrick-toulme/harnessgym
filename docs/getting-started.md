# Getting Started

The fastest way to understand HarnessGym is to:

1. run the offline demo and watch the loop,
2. read the result it produces,
3. then point it at a real task with a real runner.

## Mental model

HarnessGym runs an **iterative loop**. Each iteration is five phases:

```text
attempt → reflect → build → qualify → replay
```

- The **attempt** is a fresh agent session working your task.
- **Reflect** and **build** resume that same session to generate *one* reusable
  tool under `.harnessgym/`.
- **Qualify** copies that tool into a clean workspace and self-tests it; broken
  tools are quarantined.
- **Replay** activates the surviving tools for the next fresh session.

The source of truth is `.harnessgym/` in your workspace. Attempts come and go;
the harness accumulates. See [How It Works](how-it-works.md) for the full
walkthrough.

## The offline demo

This needs no agent account — the deterministic `fake` runner exercises the
whole loop:

```bash
harnessgym run \
  --task examples/numerical_debug_task/task.md \
  --workspace examples/numerical_debug_task \
  --iterations 2 \
  --attempt-timeout 10s \
  --build-timeout 10s \
  --runner fake
```

The fake runner intentionally blocks the first attempt, creates
`.harnessgym/tools/harnessgym_fake_probe.py`, updates the registry, starts a
fresh second attempt with that registry context, applies the known fix to
`kernel_ops.py`, runs `python verifier.py`, and records the verified result.

When it finishes, look at what it wrote:

```bash
cat examples/numerical_debug_task/.harnessgym/registry.json
ls  examples/numerical_debug_task/.harnessgym/runs/*/iterations/*/
```

## Reading a result

Every attempt writes `result.json`. The shape HarnessGym expects is:

```json
{
  "status": "solved | blocked | incomplete | tooling_built | failed",
  "verified": true,
  "summary": "what happened",
  "reflection": {
    "selected_improvement": {
      "kind": "skill | mcp | tool | verifier | fixture | docs | script | test",
      "name": "short name",
      "reason": "why this is the highest-leverage next addition",
      "target_path": ".harnessgym/<area>/..."
    }
  },
  "verification": { "status": "passed | failed | not_run", "tooling_tests": [] },
  "metrics": { "score": 0 }
}
```

HarnessGym reads `status` / `verified` to decide whether the task is solved,
`reflection.selected_improvement` to know what to build, and `metrics` to track
optimization progress.

## A real run

Point HarnessGym at your own task and a real runner. The minimum is a task file
and a workspace:

```bash
harnessgym run \
  --task task.md \
  --workspace . \
  --iterations 3 \
  --attempt-timeout 45m \
  --build-timeout 20m \
  --runner exec
```

For an open-ended optimization task, add objective scoring so HarnessGym can
checkpoint the best workspace itself — even if an attempt is killed by the
timeout mid-edit:

```bash
harnessgym run \
  --task task.md \
  --workspace . \
  --iterations 5 \
  --attempt-timeout 5m \
  --runner exec \
  --optimization-mode \
  --score-key best_cycles \
  --stop-score 1 \
  --post-attempt-command "python3 benchmark.py --json --mode final" \
  --post-attempt-score-key best_cycles
```

`--stop-score 1` is deliberately unreachable here, so all five iterations run
and the harness keeps maturing.

## Writing a good task file

A task file is just markdown. The examples that work best with HarnessGym share
a few traits:

- **An objective command.** "Run `python3 benchmark.py --json` and reduce
  `best_cycles`" gives the agent — and HarnessGym's `--post-attempt-command` —
  something concrete to score.
- **Clear edit boundaries.** "Edit `kernel.c` only; don't change
  `benchmark.py` or the tolerances" keeps comparisons honest.
- **Fast vs final modes.** A quick dev mode for iteration and a held-out final
  mode for claiming a robust improvement.
- **A nudge toward harness-building.** "If building harness improvements, prefer
  a skill plus an MCP server exposing validation, analysis, search, and
  rollback tools."

The bundled `examples/tensor_layout_pipeline_task/task.md` is a good template.

## Choosing task state

```bash
# Compound task edits across iterations (default for `run`).
harnessgym run --task task.md --workspace . --task-state continue

# Restore task files before each new iteration, keeping .harnessgym artifacts.
harnessgym run --task task.md --workspace . --task-state reset
```

Use `continue` to keep improving the same working tree. Use `reset` when you
want each fresh session to face the original task with only the *generated*
harness carried forward — which is what makes a clean before/after comparison.

## What to read next

- [How It Works](how-it-works.md) — the five phases in detail.
- [CLI Reference](cli.md) — every flag for `run` and `compare`.
- [Runners](runners/index.md) — Codex, Claude Code, fake.
- [Replay & Compare](concepts/compare.md) — measure plain vs harnessed.
- [Examples](examples/index.md) — real tasks you can run today.
