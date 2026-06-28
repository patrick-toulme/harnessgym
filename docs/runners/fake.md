# Fake & `tui-goal` runners

## `fake` — deterministic, offline

The `fake` runner (`runners.fake_runner.FakeRunner`) simulates the whole loop
with no model access and no network. It's the backend behind the install smoke
check and most of the test suite.

```bash
harnessgym run \
  --task examples/numerical_debug_task/task.md \
  --workspace examples/numerical_debug_task \
  --iterations 2 \
  --attempt-timeout 10s --build-timeout 10s \
  --runner fake
```

It scripts a realistic loop:

1. A **failed/blocked first attempt** — the task is intentionally not solved.
2. **Same-session reflection** that selects a probe tool.
3. **Artifact creation** — writes `.harnessgym/tools/harnessgym_fake_probe.py`
   and updates the registry.
4. A **fresh second attempt** that sees the generated registry context, applies
   the known demo fix to `kernel_ops.py`, runs `python verifier.py`, and records
   a verified result.

Because it's deterministic, the fake runner is ideal for:

- verifying an install without an agent account,
- exercising the orchestrator, registry, activation, and result-handling paths
  in tests,
- demoing the attempt → reflect → build → replay loop end to end in seconds.

It does **not** generate a real MCP server — it's a scaffold for the loop
mechanics, not a substitute for a real agent.

## `tui-goal` — experimental PTY backend { #tui-goal }

The `tui-goal` runner (`runners.tui_goal_runner.TuiGoalRunner`) launches an
**interactive** Codex through a pseudo-terminal and sends a real `/goal`
command for the attempt phase, then sends the reflection/build prompts to the
same process.

```bash
harnessgym run --task task.md --workspace . --runner tui-goal
```

Completion is inferred from `result.json` rather than a clean process exit,
which makes it less robust than `exec`. It exists to explore driving the
interactive Codex TUI; the **`exec` runner remains the recommended MVP path**.

[**Runners overview →**](index.md){ .md-button }
[**Getting Started →**](../getting-started.md){ .md-button }
