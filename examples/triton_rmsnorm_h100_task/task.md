# H100 Triton Fused RMSNorm Gate Optimization

Optimize the Triton fused RMSNorm + SiLU gate kernel for the lowest `best_us` from:

```bash
python3 benchmark.py --json --mode dev
```

The kernel computes:

```text
y = x * rsqrt(mean(x * x) + eps) * weight * silu(gate)
```

The objective is total best latency in microseconds across all benchmark cases. Lower is better.

You may edit:

- `kernel.py`
- `kernel_config.json`
- helper scripts or notes under `.harnessgym/` during the build phase only

Do not edit:

- `benchmark.py`
- `verifier.py`
- correctness tolerances
- benchmark cases, seeds, or task rules

Requirements:

- Run `python3 benchmark.py --json --mode dev` after meaningful changes.
- Run `python3 verifier.py --json --mode final` before claiming a robust improvement.
- Record the numeric `best_us` score in `.harnessgym/.../result.json` under both `metrics.best_us` and `metrics.score`.
- Preserve correctness: every dev/final case must pass `max_abs <= 7.5e-3`.
- Keep optimizing until the HarnessGym attempt times out, is blocked, or reaches the configured stop score.
- If running from a non-GPU workstation, use `remote_h100.py` to sync the workspace and run the command on the configured CUDA host:

```bash
python3 remote_h100.py -- python3 benchmark.py --json --mode dev
python3 remote_h100.py -- python3 verifier.py --json --mode final
```

For the remote wrapper, set:

```bash
export HARNESSGYM_GPU_HOST=user@host
export HARNESSGYM_GPU_PORT=22
export HARNESSGYM_GPU_KEY=~/.ssh/id_ed25519
```

Optimization hints:

- `num_warps`, `num_stages`, and `block_size` interact differently across 1024, 2048, 4096, and 8192-wide rows.
- The dev set is not the full objective; final mode includes held-out row counts and an 8192-wide case.
- Triton compile time is real. Good harness tooling should cache benchmark history, sweep safe config variants, rank robust winners, and avoid leaving a failing candidate in the workspace.
- For generated HarnessGym improvements, prefer a real skill plus an MCP/debug server that exposes config sweeps, trace parsing, benchmark history, rollback-safe candidate ranking, and self-tests.
