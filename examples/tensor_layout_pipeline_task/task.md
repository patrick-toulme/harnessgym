# Never-Ending Tensor Layout Pipeline Optimization

Optimize `kernel_plan.json` for the lowest `best_cycles` from:

```bash
python3 benchmark.py --json --mode dev
```

This is a custom-kernel harness task modeled after tensor layout and DMA pipeline work. The plan controls tiling, tensor layouts, vector width, accumulation, softmax strategy, DMA staging, burst size, prefetch distance, split-K, scratchpad allocation, scheduling, swizzling, and epilogue fusion. Correctness is mandatory, but the objective is open-ended: keep reducing `best_cycles`.

Requirements:
- Edit `kernel_plan.json` only when changing the kernel plan.
- Do not change `benchmark.py`, `verifier.py`, or correctness tolerances.
- Run `python3 benchmark.py --json --mode dev` after meaningful plan changes.
- Run `python3 benchmark.py --json --mode final` before claiming a robust improvement.
- Use `python3 benchmark.py --json --mode dev --trace trace.json` when you need per-case component breakdowns.
- Record the numeric `best_cycles` score in `.harnessgym/.../result.json` under both `metrics.best_cycles` and `metrics.score`.
- Keep optimizing until the HarnessGym attempt times out, is blocked, or reaches the configured stop score.
- If building harness improvements, prefer a real Codex skill plus an MCP/debug server that exposes plan validation, tensor-layout analysis, DMA descriptor synthesis, trace comparison, search/sweep, rollback-safe plan application, history comparison, and comprehensive self-tests.

The search space is intentionally interaction-heavy. Good plans usually need coordinated layout, DMA, schedule, and numerical decisions; isolated manual tweaks are often misleading.
