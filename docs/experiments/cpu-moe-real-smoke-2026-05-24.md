# CPU MoE HarnessGym Real Smoke - 2026-05-24

Workspace copy:

```text
tmp/cpu_moe_harnessgym_real_20260524T155549Z
```

Command:

```bash
PYTHONPATH=/Users/ptoulme/harnessgym/src python3 -m harnessgym.cli run \
  --task tmp/cpu_moe_harnessgym_real_20260524T155549Z/task.md \
  --workspace tmp/cpu_moe_harnessgym_real_20260524T155549Z \
  --iterations 2 \
  --attempt-timeout 5m \
  --reflection-timeout 2m \
  --build-timeout 5m \
  --runner exec \
  --post-attempt-command "python3 verifier.py --mode final --json" \
  --post-attempt-score-key best_cycles \
  --post-attempt-timeout 2m \
  --optimization-mode \
  --stop-score 850000 \
  --score-key best_cycles \
  --task-state continue \
  --harness-depth deep \
  --run-id real-smoke \
  --no-qualify-artifacts
```

Result:

- Run artifacts: `tmp/cpu_moe_harnessgym_real_20260524T155549Z/.harnessgym/runs/real-smoke`
- Baseline final `best_cycles`: `7273200`
- Iteration 1 final `best_cycles`: `5362400` (`26.27%` reduction)
- Iteration 2 final `best_cycles`: `1026800` (`85.88%` reduction)
- Post-restore final verifier rerun: passed, `best_cycles=1067068`, `max_abs=1.2085805193434718e-08`
- Final status: `tooling_built`; target `850000` was not reached in this short smoke run.

Harness behavior observed:

- Iteration 1 built `.harnessgym/skills/cpu-moe-optimizer/SKILL.md`.
- Iteration 1 built `.harnessgym/mcp/cpu_moe_harness/` with a Content-Length stdio MCP server.
- Iteration 1 built `.harnessgym/tests/test_cpu_moe_harness.py`.
- Iteration 2 started as a fresh Codex session with the generated skill and MCP active.
- Iteration 2 recorded `16` generated MCP calls across:
  `assembly_summary`, `batch_plan`, `config_sweep`, `history_compare`,
  `numerical_check`, `rank_next_experiments`, `repeat_benchmark`,
  `run_benchmark`, and `trace_routes`.
- Iteration 2 extended the MCP with `batch_plan` and `repeat_benchmark`.
- Generated tooling tests passed:
  `python3 .harnessgym/tests/test_cpu_moe_harness.py`,
  `python3 -m py_compile .harnessgym/mcp/cpu_moe_harness/server.py .harnessgym/tests/test_cpu_moe_harness.py`,
  `batch_plan_mcp_smoke`, and `repeat_benchmark_mcp_smoke`.

Implementation found by the run:

- Switched `kernel_config.json` from `route_mode=token` to `route_mode=bucketed`.
- Added NEON-assisted batched expert paths in `moe_kernel.c`.
- The remaining worst final case is `final_224x80x128_e16_adversarial`, around `1.03M` cycles on the validation rerun.

Notes:

- The score is timing-based and will vary slightly by host load; the HarnessGym run summary records `1026800`, while immediate verifier reruns observed `1033865` and post-restore observed `1067068`.
- This was a two-iteration smoke, not a full long comparison against equal-budget vanilla Codex.
