# Never-Ending CPU MoE Kernel Cycle Optimization

Optimize the CPU top-2 mixture-of-experts inference kernel for the lowest `best_cycles` from:

```bash
python3 benchmark.py --json --mode dev
```

This task models a real inference bottleneck: top-2 routing sends each token to two experts, each expert runs a small MLP, and the output is a weighted sum. The hard part is not one matrix multiply; it is the ragged routing, skewed expert buckets, weight/cache locality, SIMD-friendly inner loops, and final-mode generalization across routing distributions.

You may edit:

- `moe_kernel.c`
- `moe_kernel.h`
- `kernel_config.json`

Do not edit:

- `benchmark.py`
- `verifier.py`
- correctness tolerances
- task rules

Requirements:

- Run `python3 benchmark.py --json --mode dev` after meaningful changes.
- Run `python3 benchmark.py --json --mode final` before claiming a robust improvement.
- Use `python3 benchmark.py --json --mode dev --trace trace.json` to inspect expert load, route skew, work estimates, and per-case bottlenecks.
- Use `python3 benchmark.py --json --mode dev --assembly moe_kernel.s` when assembly/vectorization evidence would help.
- Record the numeric `best_cycles` score in `.harnessgym/.../result.json` under both `metrics.best_cycles` and `metrics.score`.
- Keep optimizing until the HarnessGym attempt times out, is blocked, or reaches the configured stop score.
- Preserve correctness: every dev/final case must pass `max_abs <= 2.5e-3`.
- If building harness improvements, prefer a real Codex skill plus an MCP/debug server that exposes route-bucket analysis, config/source variant benchmarking, assembly summaries, rollback-safe search, correctness fuzzing, history comparison, and experiment ranking.

Optimization hints:

- `route_mode=token` has simple control flow but poor expert weight locality.
- `route_mode=expert_scan` improves weight locality but can waste scans on empty experts.
- `route_mode=bucketed` can help skewed routes but has dispatch overhead.
- `hidden_tile`, `output_tile`, `input_unroll`, and `hidden_unroll` interact with register pressure and compiler vectorization.
- A dev winner can overfit uniform or Zipf routing and lose on final adversarial/bursty routes.
- Final mode includes held-out shapes and routing distributions absent from dev.

The search space is intentionally interaction-heavy. Good progress usually needs tooling that can compare route distributions, assembly/vectorization signals, config sweeps, and final-mode robustness instead of isolated manual tweaks.
