# H100 Triton RMSNorm Harness Artifacts

This directory contains the reusable `.harnessgym/` artifacts generated during the real H100 Triton RMSNorm experiments documented in:

```text
docs/experiments/h100-triton-rmsnorm-2026-05-26.md
docs/experiments/h100-triton-rmsnorm-long-2026-05-27.md
```

The bundle includes:

- `.harnessgym/skills/h100-triton-rmsnorm/SKILL.md`
- `.harnessgym/mcp/h100_triton_rmsnorm/server.py`
- `.harnessgym/mcp/h100_triton_rmsnorm/harnessgym-mcp.json`
- `.harnessgym/tests/test_h100_triton_mcp.py`

The MCP server exposes objective runs, remote H100 health checks, rollback-safe config/source/joint sweeps, guarded final verification, source diagnostics, history ranking, repeated scoring, experiment recommendations, and numerical probes. Its self-test is portable and does not require a GPU.

The current committed bundle is from the longer follow-up run, where the verified H100 score reached `99.744 us` versus a `142.848 us` baseline. The bundle qualified with one active MCP and 17 active tools.

To replay with these artifacts:

```bash
rm -rf tmp/h100_replay
mkdir -p tmp/h100_replay
cp -R examples/triton_rmsnorm_h100_task/. tmp/h100_replay/
cp -R examples/triton_rmsnorm_h100_harness_artifacts/.harnessgym tmp/h100_replay/
cd tmp/h100_replay
python3 .harnessgym/mcp/h100_triton_rmsnorm/server.py --self-test
python3 .harnessgym/tests/test_h100_triton_mcp.py
```

For real H100 scoring, set `HARNESSGYM_GPU_HOST`, `HARNESSGYM_GPU_PORT`, and `HARNESSGYM_GPU_KEY`, then run the task with `harnessgym run`.
