# H100 Triton RMSNorm Experiment

Date: 2026-05-26 local / 2026-05-27 UTC

This experiment ran HarnessGym on a real NVIDIA H100 80GB HBM3 host over SSH. Codex ran locally with the `exec` runner, while all objective GPU scoring ran on the H100 through `examples/triton_rmsnorm_h100_task/remote_h100.py`.

## Task

Template:

```text
examples/triton_rmsnorm_h100_task/
```

Objective:

```bash
python3 verifier.py --json --mode final --warmup 10 --repeats 20
```

Score key: `best_us`; lower is better.

The task optimizes a Triton fused RMSNorm + SiLU gate kernel across held-out H100 shapes, including an 8192-wide row case that punishes dev-only tuning.

## Command Run

```bash
rm -rf tmp/h100_triton_real_20260526
mkdir -p tmp/h100_triton_real_20260526
cp -R examples/triton_rmsnorm_h100_task/. tmp/h100_triton_real_20260526/

HARNESSGYM_GPU_HOST=<user@h100-host> \
HARNESSGYM_GPU_PORT=<ssh-port> \
HARNESSGYM_GPU_KEY=~/.ssh/id_ed25519 \
PYTHONPATH=src \
python3 -m harnessgym.cli run \
  --task tmp/h100_triton_real_20260526/task.md \
  --workspace tmp/h100_triton_real_20260526 \
  --iterations 2 \
  --attempt-timeout 5m \
  --reflection-timeout 3m \
  --build-timeout 5m \
  --post-attempt-command 'python3 remote_h100.py --workspace h100_triton_real_post -- python3 verifier.py --json --mode final --warmup 10 --repeats 20' \
  --post-attempt-score-key best_us \
  --post-attempt-timeout 3m \
  --score-key best_us \
  --stop-score 90 \
  --optimization-mode \
  --runner exec
```

Run id:

```text
20260527T003911Z-32309bdb
```

Run artifacts were captured under:

```text
tmp/h100_triton_real_20260526/.harnessgym/runs/20260527T003911Z-32309bdb/
```

## Results

- Baseline final score: `150.016 us`
- Best post-attempt score: `103.328 us`
- Best-score reduction: `31.1%`
- Stop score `90 us`: not reached
- Final status: `tooling_built`
- Best checkpoint restored: yes
- Post-restore verification: passed, `104.576 us` on a repeated H100 verifier run

Iteration details:

| Iteration | Attempt | Harness State | Post-attempt final score | Notes |
| --- | --- | --- | ---: | --- |
| 1 | timed out at 5m | no generated harness yet | `103.328 us` | Codex improved the kernel and HarnessGym captured the independent H100 score despite timeout. |
| 1 build | timed out at 5m, then repaired | generated skill + MCP + tests | n/a | Qualification caught a bad MCP self-test path; repair fixed it and qualification passed. |
| 2 | timed out at 5m | 1 skill, 1 MCP, 7 active tools at attempt start | `107.552 us` | Fresh Codex session used generated MCP tools 10 times; candidate regressed vs iteration 1, so best checkpoint remained iteration 1. |
| 2 build | completed | MCP expanded to 10 active tools | n/a | Qualification passed with `active_mcp_count=1`, `active_tool_count=10`. |

## Generated Harness

Committed reusable artifact bundle:

```text
examples/triton_rmsnorm_h100_harness_artifacts/.harnessgym/
```

Generated artifacts:

- `.harnessgym/skills/h100-triton-rmsnorm/SKILL.md`
- `.harnessgym/mcp/h100_triton_rmsnorm/server.py`
- `.harnessgym/mcp/h100_triton_rmsnorm/harnessgym-mcp.json`
- `.harnessgym/tests/test_h100_triton_mcp.py`

The final MCP exposes:

- `inspect_context`
- `run_objective`
- `sweep_kernel_config`
- `restore_best_checkpoint`
- `guarded_final_verify`
- `sweep_launch_overrides`
- `rank_history`
- `diagnose_source`
- `numerical_probe`
- `update_result_json`

Telemetry showed 10 successful generated MCP calls in iteration 2, including `inspect_context`, `diagnose_source`, `numerical_probe`, `run_objective`, and `rank_history`.

## Validation

Direct H100 baseline verifier before the run:

```text
status=passed, best_us=151.040, device=NVIDIA H100 80GB HBM3
```

HarnessGym baseline post-attempt verifier:

```text
status=passed, best_us=150.016, device=NVIDIA H100 80GB HBM3
```

Generated MCP self-tests after the run:

```bash
python3 tmp/h100_triton_real_20260526/.harnessgym/mcp/h100_triton_rmsnorm/server.py \
  --workspace tmp/h100_triton_real_20260526 \
  --self-test

python3 tmp/h100_triton_real_20260526/.harnessgym/tests/test_h100_triton_mcp.py
```

Result:

```text
self-test passed
Ran 10 tests in 0.291s
OK
```

## Notes

This run proves the end-to-end loop on a real GPU task:

- objective H100 scoring was captured before, during, and after HarnessGym iterations;
- a same-session reflection/build produced a real MCP harness;
- HarnessGym qualification caught and repaired a broken MCP self-test;
- the next iteration started a fresh Codex session with the generated skill and MCP active;
- generated MCP telemetry confirmed actual tool use, not just artifact activation;
- optimization checkpointing restored the best known task workspace after a later regression.

The run did not prove that the generated harness beat the iteration-1 no-harness candidate on this short two-iteration budget. It did prove the real H100 loop works and that the harness generated complex, qualified, reusable tooling.
