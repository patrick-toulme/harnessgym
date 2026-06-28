# Never-Ending C Flash Attention CPU Kernel Optimization

Optimize `flash_attention_forward` in `kernel.c` for the lowest reported CPU cycle count.

This is a CPU proxy for custom Flash Attention kernel work. Correctness is mandatory, but the objective is open-ended: reduce `best_cycles` from the benchmark as much as possible.

Requirements:
- Preserve the public C function signature in `kernel.h`.
- Edit `kernel.c` only unless you are adding local notes or harness artifacts.
- Do not change `benchmark.c`, `benchmark.py`, `kernel.h`, or the correctness tolerances.
- Run `python3 benchmark.py --json` after every meaningful implementation change.
- Record the numeric `best_cycles` score in `.harnessgym/.../result.json` under both `metrics.best_cycles` and `metrics.score`.
- Keep optimizing until the HarnessGym attempt times out, is blocked, or reaches the configured stop score.
- If building harness improvements, prefer a real Codex skill plus an MCP/debug server that exposes compile, benchmark, compare, and kernel invariant helpers.
