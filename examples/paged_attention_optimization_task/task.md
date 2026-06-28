# Never-Ending Kernel Optimization Task

Optimize `paged_attention_kernel` in `kernel.py`.

This is a CPU proxy for custom paged-attention decode kernel work. The task is intentionally open-ended: correctness is mandatory, but the real objective is reducing benchmark time as far as possible.

Requirements:
- Preserve the public function name and signature.
- Use NumPy only.
- Maintain exact correctness against the verifier's randomized reference cases.
- Optimize the benchmark score reported by `python3 benchmark.py`.
- After each meaningful change, run `python3 benchmark.py --json` and record the numeric `best_ms` score in `.harnessgym/.../result.json` under `metrics.best_ms` and `metrics.score`.
- Do not stop just because the verifier passes; keep optimizing until the time budget is exhausted, blocked, or the configured HarnessGym stop score is reached.
- If building harness improvements, prefer a real Codex skill plus an MCP/debug server that exposes benchmark, compare-case, and invariants helpers.
