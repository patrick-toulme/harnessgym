# Tensor Layout Qualification Experiment

This note records the real Codex experiment used to validate the HarnessGym
artifact qualification work from commit `59ebaa1`.

## Objective

Prove that HarnessGym can:

- run a hard optimization task with real `codex exec` sessions;
- build reusable repo-local harness artifacts, including an MCP server;
- qualify those artifacts in a fresh copied workspace before reuse;
- replay the task with and without the generated harness under the same attempt
  budget; and
- show a meaningful improvement when the generated harness is available.

The task was `examples/tensor_layout_pipeline_task`, a synthetic tensor-layout
pipeline optimization problem scored by `benchmark.py --json --mode final`.
Lower `best_cycles` is better.

## Generation Run

Workspace:

```bash
tmp/tensor_layout_qual_exp
```

Run artifacts:

```bash
tmp/tensor_layout_qual_exp/.harnessgym/runs/<run-id>
```

Command:

```bash
PYTHONPATH=src python3 -m harnessgym.cli run \
  --task task.md \
  --workspace tmp/tensor_layout_qual_exp \
  --iterations 3 \
  --attempt-timeout 5m \
  --reflection-timeout 3m \
  --build-timeout 5m \
  --runner exec \
  --optimization-mode \
  --score-key best_cycles \
  --stop-score 1 \
  --task-state continue \
  --harness-depth deep \
  --post-attempt-command "python3 benchmark.py --json --mode final" \
  --post-attempt-score-key best_cycles \
  --post-attempt-timeout 2m \
  --artifact-repair-attempts 1
```

Results:

| Metric | Value |
| --- | ---: |
| Baseline final score | 33,975,173 |
| Iteration 2 score with active MCP tools | 1,526,080 |
| Final best score | 1,495,982 |
| Final relative cycle reduction | 95.5968% |
| Final cycle ratio | 22.71x lower |

Generated harness artifacts included:

- `.harnessgym/skills/tensor-plan-optimizer/SKILL.md`
- `.harnessgym/mcp/tensor-plan-server/tensor_plan_server.py`
- `.harnessgym/mcp/tensor-plan-server/harnessgym-mcp.json`
- `.harnessgym/fixtures/tensor_plan_attempt1_best.json`
- `.harnessgym/tests/test_tensor_plan_mcp.py`

The first generated MCP server qualified in a fresh copied workspace with
1 active MCP and 9 active tools. Later iterations extended the same MCP to
15 active tools.

## A/B Compare Run

Compare output:

```bash
tmp/tensor_layout_qual_compare
```

Report:

```bash
tmp/tensor_layout_qual_compare/compare_report.json
```

Command:

```bash
PYTHONPATH=src python3 -m harnessgym.cli compare \
  --workspace-template examples/tensor_layout_pipeline_task \
  --task task.md \
  --artifact-source tmp/tensor_layout_qual_exp/.harnessgym \
  --output-dir tmp/tensor_layout_qual_compare \
  --trials 1 \
  --iterations 1 \
  --attempt-timeout 5m \
  --runner exec \
  --score-key best_cycles \
  --stop-score 1 \
  --task-state continue \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles \
  --post-timeout 2m \
  --overwrite
```

Both trials received one 5-minute attempt. The harnessed trial required active
generated harness tooling and passed that validation.

| Group | Attempt Budget | Active MCPs | Active Tools | Final Score |
| --- | ---: | ---: | ---: | ---: |
| Plain Codex | 5m | 0 | 0 | 33,975,173 |
| Harnessed Codex | 5m | 1 | 15 | 1,495,982 |

Harnessed replay result:

- 95.60% lower `best_cycles` than plain Codex.
- 22.71x lower `best_cycles`.
- Objective verifier completed successfully in both trials.

Active MCP tools in the harnessed replay:

- `apply_best_verified`
- `apply_candidate`
- `benchmark_plan`
- `bounded_exhaustive_search`
- `candidate_diff`
- `compare_history`
- `export_candidate_fixture`
- `local_neighborhood_search`
- `numerical_check`
- `rank_next_experiments`
- `resume_search_history`
- `run_objective`
- `search_plans`
- `trace_summary`
- `validate_plan`

## Notes

- This was one A/B trial, not a statistical benchmark.
- Iteration 2 exposed a real edge case: source-workspace activation can diverge
  from fresh qualification when generated self-tests depend on mutable task
  state. Final fresh qualification and the A/B harness replay were clean.
- The compare gate now rejects harnessed trials when required generated MCP
  tools are unavailable, which protects against accidental "harnessed" runs
  that did not actually activate the harness.
- The run and compare artifacts are under `tmp/` and are intentionally not
  committed.
