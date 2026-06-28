---
name: tensor-plan-optimizer
description: Optimize HarnessGym tensor-layout kernel_plan.json tasks with a generated MCP server for plan validation, dev/final benchmarking, trace analysis, rollback-safe search, candidate application, history comparison, and experiment ranking. Use when a task asks to minimize best_cycles for benchmark.py or tune tensor layout/DMA/pipeline knobs under .harnessgym.
---

# Tensor Plan Optimizer

Use the MCP suite first:

- Manifest: `.harnessgym/mcp/tensor-plan-server/harnessgym-mcp.json`
- Server: `.harnessgym/mcp/tensor-plan-server/tensor_plan_server.py`
- Smoke test: `python3 .harnessgym/mcp/tensor-plan-server/tensor_plan_server.py --self-test`
- Codex exec helper: `python3 .harnessgym/runtime/mcp_call.py --server tensor-plan-server --tool run_objective --arguments '{"mode":"dev"}'`

For Codex exec, use the workspace-local helper above if native MCP tools are not visible. Do not write a one-off JSON-RPC client or launch the server directly; the helper calls through HarnessGym telemetry so the run records real generated-tool usage in `.harnessgym/mcp_calls.jsonl`.

## Workflow

1. Call `run_objective` in `dev` mode to confirm the current score.
2. Call `trace_summary` on the current plan to identify bank conflicts, DMA penalties, spill pressure, scratchpad pressure, and synergies.
3. Call `resume_search_history` to find the best fixtures/history seeds before repeating any sweep.
4. Call `local_neighborhood_search` with `strategy=checkpoint` around the current best fixture or plan. This covers single, pair, and bounded grid mutations and is the fastest way to reproduce the iteration-2 local-search win.
5. Call `bounded_exhaustive_search` with `profile="iteration3_dev_core"` and, if time permits, `profile="iteration3_layout_relaxed"`. These profiles encode the manual iteration-3 sweeps that found no better dev plan than the iteration-2 fixture while checking dev winners against final mode.
6. Call `search_plans` with `strategy=quick` for broader randomized exploration. Use `strategy=focused` only when there is time for a broader deterministic sweep around the best discovered region.
7. Call `candidate_diff` and `benchmark_plan` with `modes=["dev","final"]` for promising candidates. `dev` is the fast objective; `final` is the authoritative held-out comparison.
8. Call `apply_best_verified` or `apply_candidate` with `dry_run=true` first. Only use `dry_run=false` after the candidate improves `dev`, passes `final`, and preserves correctness.
9. Call `export_candidate_fixture` for a verified improvement that should seed the next attempt.
10. Call `compare_history` before final reporting so the best explored candidate is not lost.

## Task Invariants

- Edit `kernel_plan.json` only when changing the kernel plan.
- Do not edit `benchmark.py`, `verifier.py`, or correctness tolerances.
- Objective metric: `best_cycles`; lower is better.
- Fast iteration mode: `python3 benchmark.py --json --mode dev`.
- Authoritative comparison: `python3 benchmark.py --json --mode final`.
- Correctness is driven by `max_abs <= 2.5e-4` in both modes.

## Attempt-1 Findings

Successful pattern:

- `q_layout=swizzled_mn`, `k_layout=swizzled_nk`, `v_layout=swizzled_kd`, `o_layout=streamed`
- `vector_width=16`, `accum=tree`, `softmax=online_renorm`
- `schedule=persistent`, `dma_stages=3`, `prefetch_distance=2 or 3`
- `swizzle=tensorcore`, `epilogue=fused_scale_mask`, `scratchpad_kb=128`

Known traps:

- Baseline row-major layouts are extremely slow despite passing correctness.
- `approx_poly` can overfit dev and fail or degrade final dim-128 cases when paired with high-error accumulation choices.
- `burst=256` helps some dense cases but wastes cycles on sparse masks; compare against `burst=128`.
- Larger tiles and more warps can lose to spill pressure. Inspect `register_pressure` before applying.
- `split_k` helps larger sequences but can hurt the 192-token dev case; use final mode before keeping it.

The fixture `.harnessgym/fixtures/tensor_plan_attempt1_best.json` stores the best candidate found before the timed-out attempt ended. Treat it as a starting point, not a proof of global optimality.

## Attempt-2 Findings

Useful local-search pattern:

- Start from `tensor_plan_attempt1_best`.
- Keep `block_m=48`, `block_n=96`, `block_k=64`, `vector_width=16`.
- Try `num_warps=6`, `epilogue=fused_scale`, and `scratchpad_kb=96` together.
- Preserve the swizzled/tensorcore/streamed layout family, `accum=tree`, `softmax=online_renorm`, `dma_stages=3`, `prefetch_distance=3`, `dma_burst=128`, `split_k=1`, and `schedule=persistent`.

The fixture `.harnessgym/fixtures/tensor_plan_iteration2_neighborhood_best.json` captures this verified seed. It observed `dev best_cycles=262005` and `final best_cycles=1495982` in the deterministic benchmark model.

When time is short, prefer:

1. `local_neighborhood_search(fixture="tensor_plan_iteration2_neighborhood_best", strategy="checkpoint", include_final=true)`
2. `bounded_exhaustive_search(profile="iteration3_dev_core", max_evals=1814400, include_final=true)`
3. `candidate_diff` against the current plan or fixture
4. `apply_best_verified(dry_run=true)` before mutating `kernel_plan.json`

## Attempt-3 Findings

Actual iteration-3 search applied the iteration-2 fixture to the workspace and verified it at `dev best_cycles=262005` and `final best_cycles=1495982`. Two bounded exhaustive sweeps were run manually:

- `iteration3_dev_core`: 1,814,400 candidates in the swizzled/tensorcore/streamed layout family with numeric DMA/tile variations.
- `iteration3_layout_relaxed`: 1,990,656 candidates around the winning tile region with near-optimal layout, schedule, and epilogue alternatives.

Both sweeps ranked the same plan first: `block_m=48`, `block_n=96`, `block_k=64`, `num_warps=6`, `vector_width=16`, swizzled Q/K/V, streamed O, `tree`, `online_renorm`, `dma_stages=3`, `prefetch_distance=3`, `dma_burst=128`, `split_k=1`, `tensorcore`, `persistent`, `fused_scale`, `scratchpad_kb=96`. The final-mode bottleneck remains `final_384x128_dense`; next exploration should explicitly test final-hotspot-friendly `block_k=96`, `split_pipeline`, and `split_k=2` variants without regressing dev correctness.

## TODO

- Add HarnessGym auto-registration glue if future runners do not discover `.harnessgym/mcp/tensor-plan-server/harnessgym-mcp.json` automatically.
- Add a stronger final-mode repeated-evaluation tool if the benchmark becomes noisy or gains additional held-out seeds.
- Consider adding final-mode repeated evaluation if future benchmarks introduce stochastic timing noise; current scores are deterministic.

## Result Recording

Before ending an optimization attempt, run the final benchmark and write the chosen numeric `best_cycles` to both `metrics.best_cycles` and `metrics.score` in the iteration result JSON. Also record generated tooling under `used_harness_artifacts` or `used_harness_tools` when applicable.
