---
name: h100-triton-rmsnorm
description: Use for the H100 Triton fused RMSNorm + SiLU gate optimization task. Provides the workflow and MCP tooling for objective runs, rollback-safe config sweeps, source diagnostics, benchmark history, and final held-out verification.
---

# H100 Triton RMSNorm Harness

Use this skill at the start of a fresh attempt on this workspace.

## First Actions

1. Inspect active context and prior results with the MCP tool `inspect_context`.
2. If `HARNESSGYM_GPU_HOST` is set or local CUDA is uncertain, run `remote_health_check` before any objective sweep.
   - It checks SSH reachability, remote scratch space, and `nvidia-smi` visibility without starting a tar sync.
   - If it reports `failure_stage="ssh"`, stop and record the infrastructure blocker instead of spending time on benchmark commands that will only surface `tar: Write error`.
3. Use `run_objective` for fast iteration:
   - `mode="dev"`, `verifier=false` runs `benchmark.py --json --mode dev`.
   - Objective metric is `best_us`; lower is better.
4. Use `run_objective` for authoritative comparison:
   - `mode="final"`, `verifier=true` runs `verifier.py --json --mode final`.
   - Preserve correctness: every case must pass `max_abs <= 7.5e-3`.
5. Before ending on a risky candidate, call `guarded_final_verify` with `restore_on_regression=true`.
   - It compares against the best checkpoint score and restores `kernel.py` / `kernel_config.json` if the candidate regresses.
6. Record candidate runs in history and compare with `rank_history`.
7. For combined source and launch search, prefer `joint_source_launch_search` before manual patching.
   - Start with a small dev-mode filter, then confirm top candidates with final verifier mode.
   - It can search `current`, exact SiLU (`exp`, `sigmoid`, `exp2`), and rational SiLU variants across combined 2048/4096 launch overlays while restoring files by default.
8. For source-level math variants, prefer `sweep_silu_variants` over manual patching.
   - It can test `exp`, `sigmoid`, and `exp2` SiLU forms while preserving/restoring `kernel.py`.
   - Cross it with a small list of launch overlays when investigating the 8192 case.
9. For approximate SiLU math, do not edit `kernel.py` by hand first.
   - Run `probe_silu_approximations` with `mode="all"` to check toy plus dev/final-shape proxy tolerance.
   - Then run `sweep_silu_approximations`; it is rollback-safe and skips variants that fail the numerical prefilter.
10. Before trusting a borderline winner, run `repeat_objective` in final verifier mode.
   - Use `median_score` and per-case spreads to avoid keeping one lucky timing sample.

## Task Invariants

- Kernel computes `y = x * rsqrt(mean(x*x) + eps) * weight * silu(gate)`.
- Dev cases are fixed seeds/shapes: `(1024,1024)`, `(768,2048)`, `(384,4096)`.
- Final held-out cases are fixed seeds/shapes: `(1536,1024)`, `(896,2048)`, `(448,4096)`, `(192,8192)`.
- Final includes an 8192-wide case; optimizing only dev overfits.
- Valid score keys are `best_us` and `score`; both should mirror the final objective when updating `result.json`.
- On non-GPU workstations, use `remote_h100.py`; the MCP tool chooses remote automatically when `HARNESSGYM_GPU_HOST` is present.

## Patterns From Iteration 1

- Replacing explicit `gate / (1 + exp(-gate))` with `gate * tl.sigmoid(gate)` preserved correctness and improved final latency.
- A single global `num_warps=1` left the 8192-wide held-out case near `76 us`; multi-warp launch choices brought it near `23-25 us`.
- Per-case timing is noisy. Prefer repeated final-mode verification or `rank_history` evidence over one isolated sweep winner.
- A multi-row-per-program experiment looked plausible but regressed post-attempt timing; treat row-grouping changes as risky and verify with final mode.

## Patterns From Iteration 2

- A one-process helper sweep with short repeats produced misleading absolute timings and selected a launch config that regressed final score from about `94.7 us` to `107.6 us`.
- Treat launch override sweeps as exploratory until confirmed by `verifier.py --json --mode final`.
- Use `sweep_launch_overrides` only when `source_supports_launch_overrides=true`; otherwise add a tunable launch hook or make explicit source variants before sweeping.
- The objective is total `best_us`, mirrored as `score`; lower is better. Do not optimize only dev shapes because final adds `(192,8192)`.
- A hand-written rational SiLU approximation (`rational_m3n2`) passed fixed-shape tolerance probes but regressed final latency to about `107.5 us`, mainly hurting the 8192 case. Approximate math needs both numerical prefiltering and final-mode timing before it is kept.
- The new approximation tools know `rational_m2n2`, `rational_m3n2`, and `rational_m3n3`. Treat them as candidates, not defaults; exact `exp` plus launch overrides remained more competitive in this attempt.

## Patterns From Iteration 1 In This Workspace

- A per-dimension 8192 launch override improved final from `142.848 us` to about `104.832 us`, but did not hit the `90 us` target.
- The robust post-attempt candidate used global `num_warps=1`, `num_stages=4`, and 8192 override `num_warps_8192=32`, `num_stages_8192=1`, `rows_per_program_8192=1`.
- One-off final sweeps saw lower noisy samples near `98 us`, but post-attempt verification settled higher. Do not keep a launch/source variant without repeated final-mode evidence.
- Manual SiLU source edits consumed time. Use `sweep_silu_variants` next to compare `gate / (1 + exp(-gate))`, `gate * tl.sigmoid(gate)`, and `exp2`-based SiLU under the best 8192 launch overlays.

## Patterns From Iteration 3

- The post-attempt verified score improved to `99.744 us`, still above the `90 us` target.
- The kept config stayed conservative: global `num_warps=1`, `num_stages=4`, `rows_per_program=1`, plus 1024 override `(rows_per_program=4, num_warps=4, num_stages=3)` and 8192 override `(rows_per_program=1, num_warps=32, num_stages=1)`.
- Single-dimension final sweeps for 2048/4096 produced low samples near `100 us`, but combined overlays were noisy and did not survive post-attempt verification.
- `sweep_silu_variants` could not start from a rational-only source because it only recognized exact SiLU expressions. Use `joint_source_launch_search`; it renders exact or rational variants from either exact or rational source states.
- Treat dev as a compile/filter pass. Any candidate near `97-100 us` needs repeated final verifier confirmation because one-off best timings moved several microseconds.

## Patterns From Iteration 4

- The attempt was blocked by remote infrastructure: `root@64.247.201.35:15548` refused SSH, and `remote_h100.py` surfaced the failure as `tar: Write error` during sync.
- Run `remote_health_check` at the start of the next fresh attempt. It separates SSH, GPU, and remote scratch-space failures before launching expensive search or verifier commands.
- The workspace stayed at the iteration-3 verified checkpoint (`99.744 us`); do not infer a new kernel result from this blocked attempt.

## MCP Tools

The paired MCP server lives at `.harnessgym/mcp/h100_triton_rmsnorm/`.

- `inspect_context`: summarize activation, cases, prior result, baseline, source hashes.
- `remote_health_check`: preflight remote SSH reachability, scratch disk space, and `nvidia-smi` visibility before sync/benchmark runs.
- `run_objective`: run dev/final benchmark or verifier locally/remote and append history.
- `sweep_kernel_config`: rollback-safe sweep of `kernel_config.json` variants.
- `sweep_launch_overrides`: rollback-safe generated sweep of per-dimension launch override configs for tunable kernels.
- `sweep_silu_variants`: rollback-safe source sweep of equivalent SiLU formulas, optionally crossed with launch/config overlays.
- `probe_silu_approximations`: deterministic CPU-side toy and dev/final-shape proxy tolerance checks for rational SiLU approximations.
- `sweep_silu_approximations`: rollback-safe source sweep of rational SiLU approximations with numerical prefiltering before benchmark/verifier runs.
- `joint_source_launch_search`: rollback-safe cross-product search over exact/rational/current SiLU source variants and combined launch overlays, with optional repeated final confirmation.
- `repeat_objective`: repeated dev/final objective runner that reports min/median/max score and per-case timing spread.
- `recommend_next_experiments`: history and held-out-shape summary with suggested next MCP commands.
- `guarded_final_verify`: final/dev objective run that restores the best checkpoint on regression.
- `restore_best_checkpoint`: manually restore rollback-safe mutable files from the best checkpoint.
- `rank_history`: rank recorded objective runs and expose regressions.
- `diagnose_source`: inspect `kernel.py` for implementation patterns and source diff.
- `numerical_probe`: deterministic CPU-side tolerance checks for SiLU/RMSNorm formulas.
- `update_result_json`: write score, verification, and reflection fields into the latest or specified result JSON.

Run the server self-test before relying on the tools:

```bash
python3 .harnessgym/mcp/h100_triton_rmsnorm/server.py --self-test
python3 .harnessgym/tests/test_h100_triton_mcp.py
```

## TODO

- Consider having `run_objective` call `remote_health_check` automatically when remote execution is selected, with a short cache TTL so repeated sweeps do not pay the SSH preflight cost for every candidate.
- Add a Triton IR/PTX diagnostic tool if later attempts need compiler-level evidence for why launch variants regress.
- Add optional PTX/SASS extraction for exact-vs-approx SiLU variants to explain why arithmetic approximations can slow the 8192 held-out case.
