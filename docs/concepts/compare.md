# Replay & Compare

`harnessgym run` *generates* a harness. `harnessgym compare` *measures* it:
it replays plain attempts against harnessed attempts from a clean template,
under equal budgets, and writes a `compare_report.json` you can trust.

## The shape of a comparison

```bash
harnessgym compare \
  --workspace-template examples/cpu_attention_autotune_task \
  --task task.md \
  --artifact-source tmp/<run>/.harnessgym \
  --output-dir tmp/cpu_attention_autotune_compare \
  --trials 1 \
  --iterations 5 \
  --attempt-timeout 2m \
  --runner exec \
  --score-key best_cycles \
  --stop-score 1 \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles
```

For each trial, compare:

1. **Copies the clean `--workspace-template`** into a fresh trial directory
   (`copy_workspace_template`). Plain and harnessed trials start from identical
   task files.
2. For **harnessed** trials, copies in the `.harnessgym/` bundle from
   `--artifact-source` (`copy_harness_artifacts`); plain trials get nothing.
3. Runs an **attempt-only** replay (no reflect/build) with the configured
   budget.
4. Optionally runs `--post-command` for an objective held-out score.
5. Summarizes MCP telemetry for the trial.

Because the only difference between the arms is the presence of the generated
harness, a score delta is attributable to the harness — not to a different
starting point.

## The report

`compare_report.json` records each trial and a per-group summary. Each trial
carries the attempt durations, the post score (and whether it's valid), and the
MCP telemetry:

```json
{
  "summary": {
    "plain":     { "count": 1, "post_score": { "best": 1008322, "valid_count": 1, "invalid_count": 0 }, ... },
    "harnessed": { "count": 1, "post_score": { "best": 130223,  "valid_count": 1, "invalid_count": 0 },
                   "mcp_call_count": { "median": 14 }, "mcp_called_tools": ["autotune", "evaluate_final", ...] }
  },
  "trials": [
    { "group": "harnessed", "post_score": 130223, "post_valid": true,
      "mcp_call_count": 14, "mcp_called_tools": ["..."], ... }
  ]
}
```

## Validity rules: what stops you fooling yourself

A naive A/B is easy to game. Compare encodes several guards so a broken or
unused harness can't be mistaken for a win:

- **Failed post commands are worst-case, not missing.** If `--post-command`
  fails, the trial is recorded with `post_valid: false`,
  `post_invalid_reason`, and counted as an invalid worst-case outcome
  (`post_treated_as_worst`) — never silently dropped.
- **Harnessed trials must actually activate a tool.** By default
  (`--require-active-harness`) a harnessed trial is invalid for comparison
  unless the copied artifacts activate at least one generated MCP tool. So a
  broken bundle can't masquerade as a harness win. Use
  `--no-require-active-harness` only for smoke tests that intentionally copy
  docs/scripts.
- **Optionally, the agent must *call* a tool.** With
  `--require-harness-tool-use`, a harnessed trial counts only if at least one
  generated MCP `tools/call` is recorded in the telemetry — activation alone is
  not enough. This is the strongest evidence: the agent used the tool, and
  HarnessGym has the log.

## Reading the result honestly

When both arms time out at the same cap, a lower harnessed score proves a
**better reached score in the same model-time budget**, not faster wall-clock
completion. The [results](../results.md) page states which kind of win each
number is. One or a few trials is engineering evidence, not a statistical
claim; raise `--trials` for more samples.

## A scripted replay

Some examples ship a helper that first checks the runner can make model calls,
then audits the resulting report for valid plain/harnessed trials, valid post
scores, and the expected active tool count — for example
`examples/tensor_layout_harness_artifacts/run_claude_compare.sh`, which defaults
to `REQUIRE_HARNESS_TOOL_USE=1`.

[**MCP Telemetry →**](telemetry.md){ .md-button }
[**Results →**](../results.md){ .md-button }
