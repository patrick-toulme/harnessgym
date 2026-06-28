# C++ Stencil Kernel Optimization Task

Optimize `kernel.cpp` for the lowest `best_cycles` reported by:

```bash
python3 benchmark.py --json --mode dev
```

The kernel applies a clamped five-point stencil to a float grid. Correctness is mandatory: keep `max_abs <= 2.5e-5` on all dev and final cases.

Rules:

- Edit `kernel.cpp` and `kernel.h` only when changing the kernel implementation.
- Do not change `benchmark.py` or `verifier.py` to improve the score.
- Do not tamper with Python timing, Python C APIs, dynamic symbol lookup, process clocks, or benchmark internals from the kernel.
- Use dev mode for fast iteration.
- Run final mode before claiming a robust improvement:

```bash
python3 benchmark.py --json --mode final
python3 verifier.py
```

Useful diagnostics:

```bash
python3 benchmark.py --json --mode dev --trace trace.dev.json --assembly .harnessgym_build/kernel.s
```

The assembly summary reports instruction-line counts, branch mentions, vector-register mentions, FMA mentions, and load/store mentions. A good harness for this task should expose compiler artifacts, numerical checks, rollback-safe variants, benchmark history, and final-mode regression comparison.

The benchmark verifies every timed repeat with varied input buffers and includes guard cases with alternate alpha/seed values. A valid optimization must compute the stencil for each call, not cache benchmark shapes or modify timers.
