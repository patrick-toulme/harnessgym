# Long H100 Triton RMSNorm Harness Run

Date: 2026-05-26 local / 2026-05-27 UTC

This follow-up experiment ran HarnessGym for a longer budget on the real NVIDIA H100 80GB HBM3 Triton RMSNorm task. The run started from the previously committed H100 task template plus the generated `.harnessgym/` artifact bundle from the first verified experiment.

## Task

Template:

```text
examples/triton_rmsnorm_h100_task/
```

Seeded harness bundle:

```text
examples/triton_rmsnorm_h100_harness_artifacts/.harnessgym/
```

Objective:

```bash
python3 verifier.py --json --mode final --warmup 20 --repeats 50
```

Score key: `best_us`; lower is better.

## Command Run

```bash
rm -rf tmp/h100_triton_harness_long_20260527
mkdir -p tmp/h100_triton_harness_long_20260527
cp -R examples/triton_rmsnorm_h100_task/. tmp/h100_triton_harness_long_20260527/
cp -R examples/triton_rmsnorm_h100_harness_artifacts/.harnessgym tmp/h100_triton_harness_long_20260527/

HARNESSGYM_GPU_HOST=<user@h100-host> \
HARNESSGYM_GPU_PORT=<ssh-port> \
HARNESSGYM_GPU_KEY=~/.ssh/id_ed25519 \
PYTHONPATH=src \
python3 -m harnessgym.cli run \
  --task tmp/h100_triton_harness_long_20260527/task.md \
  --workspace tmp/h100_triton_harness_long_20260527 \
  --iterations 4 \
  --attempt-timeout 8m \
  --reflection-timeout 4m \
  --build-timeout 6m \
  --post-attempt-command 'python3 remote_h100.py --workspace h100_triton_harness_long_post -- python3 verifier.py --json --mode final --warmup 20 --repeats 50' \
  --post-attempt-score-key best_us \
  --post-attempt-timeout 4m \
  --score-key best_us \
  --stop-score 90 \
  --optimization-mode \
  --runner exec
```

Run id:

```text
20260527T014929Z-3b19cc76
```

Run artifacts were captured under:

```text
tmp/h100_triton_harness_long_20260527/.harnessgym/runs/20260527T014929Z-3b19cc76/
```

## Results

- Baseline final score: `142.848 us`
- Best verified post-attempt score: `99.744 us`
- Best-score reduction from baseline: `30.2%`
- Improvement over the previous committed H100 best (`103.328 us`): `3.5%`
- Improvement over this run's iteration-1 score (`104.832 us`): `4.9%`
- Stop score `90 us`: not reached
- Final status: `tooling_built`
- Best checkpoint restored: yes, iteration 3
- Final post-restore verifier: failed because the H100 SSH endpoint refused connections after iteration 4

Iteration details:

| Iteration | Attempt | Active tools | MCP calls | Post-attempt final score | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | timed out at 8m | 10 | 17 | `104.832 us` | Used the seeded MCP and built source-variant/repeated-scoring tools. |
| 2 | timed out at 8m | 13 | 28 | `107.488 us` | Built approximation-aware SiLU probes and rollback-safe approximate-source sweeps. |
| 3 | timed out at 8m | 15 | 45 | `99.744 us` | Used generated approximation/search tooling and built joint source-plus-launch search. |
| 4 | completed, infrastructure-blocked | 16 | 53 | n/a | H100 SSH began refusing connections; build added remote health preflight tooling. |

The best in-session held-out verifier sample observed during iteration 4 was `97.664 us`, but it was not independently post-attempt verified before the H100 endpoint failed. The trusted best score for this experiment is therefore `99.744 us`.

## Generated Harness

The committed reusable artifact bundle was updated at:

```text
examples/triton_rmsnorm_h100_harness_artifacts/.harnessgym/
```

The final MCP exposes 17 tools:

- `inspect_context`
- `remote_health_check`
- `run_objective`
- `sweep_kernel_config`
- `restore_best_checkpoint`
- `guarded_final_verify`
- `sweep_launch_overrides`
- `sweep_silu_variants`
- `probe_silu_approximations`
- `sweep_silu_approximations`
- `joint_source_launch_search`
- `repeat_objective`
- `recommend_next_experiments`
- `rank_history`
- `diagnose_source`
- `numerical_probe`
- `update_result_json`

The new capabilities added during this longer run were:

- rollback-safe exact SiLU source variant search;
- repeated objective scoring and next-experiment ranking;
- deterministic rational SiLU numerical probes with toy and shape-proxy cases;
- rollback-safe approximate SiLU benchmark sweeps;
- joint source-plus-launch search;
- remote H100 health preflight to classify SSH, GPU, and scratch-space failures before tar sync.

## Artifact Qualification

HarnessGym qualified the generated artifact bundle after every build phase in a fresh workspace:

| Iteration | Active MCPs | Active tools | Status |
| --- | ---: | ---: | --- |
| 1 | 1 | 13 | passed |
| 2 | 1 | 15 | passed |
| 3 | 1 | 16 | passed |
| 4 | 1 | 17 | passed |

The final generated MCP tests were also run directly outside the orchestrator:

```bash
cd tmp/h100_triton_harness_long_20260527
python3 .harnessgym/tests/test_h100_triton_mcp.py
```

Result:

```text
Ran 22 tests in 0.585s
OK
```

## Notes

This run showed the harness becoming useful across more than two turns. The seeded harness found a strong first score, then the generated approximation/search tooling helped a later fresh session reach a lower independently verified score. The final iteration did not score because of infrastructure, but it still produced a practical harness improvement: `remote_health_check`.

The result should not be read as a final kernel optimum. It proves that the HarnessGym loop can keep improving the reusable harness and can drive a lower verified score on a real H100 optimization task, while preserving checkpoints and refusing to trust unverified/noisy samples.
