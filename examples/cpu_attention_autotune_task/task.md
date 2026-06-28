# Never-Ending CPU Attention Kernel Autotuning

Optimize `kernel_config.json` for the lowest `best_cycles` from:

```bash
python3 benchmark.py --json --mode dev
```

This is a CPU custom-kernel autotuning proxy. The config controls tile sizes, vector width, unroll factors, prefetching, pipeline depth, memory layout, accumulator strategy, exp approximation, store strategy, packing, and split-K. Correctness is mandatory, but the task is open-ended: keep reducing `best_cycles`.

Requirements:
- Edit `kernel_config.json` only when changing the kernel configuration.
- Do not change `benchmark.py`, `verifier.py`, or the correctness tolerances.
- Run `python3 benchmark.py --json --mode dev` after meaningful config changes.
- Run `python3 benchmark.py --json --mode final` before claiming a robust improvement.
- Record the numeric `best_cycles` score in `.harnessgym/.../result.json` under both `metrics.best_cycles` and `metrics.score`.
- Keep optimizing until the HarnessGym attempt times out, is blocked, or reaches the configured stop score.
- If building harness improvements, prefer a real Codex skill plus an MCP/debug server that exposes config validation, dev/final evaluation, rollback-safe search, benchmark-history comparison, and experiment ranking.

The search space is intentionally large and interaction-heavy. Manual one-off tweaks are usually worse than evidence-driven search.
