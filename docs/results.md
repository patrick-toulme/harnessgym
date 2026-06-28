# Results

These are real validation runs preserved in the repository. Each one ran the
full HarnessGym path — generate an artifact, qualify it in a fresh workspace,
activate it, let the agent actually call it through MCP, and score the result
against a no-harness baseline.

!!! note "What these numbers are"
    Engineering evidence, not a statistical claim. Each result is one (or a few)
    real Codex/Claude runs on one machine. They prove the *mechanism* works
    end to end: a generated, qualified, activated harness changes what a fresh
    session reaches in the same model-time budget. Repeat runs vary.

Every comparison below uses **equal attempt budgets** for the plain and
harnessed arms. Where both arms time out, the result proves a *better reached
score in the same budget*, not lower wall-clock completion.

---

## Tensor layout pipeline

The hardest bundled CPU-style task. The agent optimizes `kernel_plan.json`,
which controls tiling, tensor layouts, vector width, softmax strategy,
accumulation, DMA staging, prefetch distance, descriptor burst size, split-K,
scratchpad, scheduling, swizzling, and epilogue fusion. The benchmark exposes
fast dev cases, held-out final cases, and detailed trace JSON.

A real five-iteration run generated a tensor-layout skill plus an MCP server
with focused search, off-spine search, candidate proof, deadline-aware apply,
fast focused apply, history, final gating, and self-tests.

| Mode | Before | After | Reduction |
| --- | --- | --- | --- |
| Dev | 8,348,345 | 262,005 | — |
| **Final (held-out)** | **33,975,173** | **1,495,982** | **95.60%** |

This task is the recommended performance-proof when evaluating whether the
harness helps: progress depends on inspecting trace/layout/descriptor
interactions and automating sweeps, not on one obvious manual rewrite.

---

## CPU attention autotune

A deterministic CPU custom-kernel autotuning proxy: the agent edits only
`kernel_config.json`, verifies correctness, and optimizes `best_cycles` across
fast dev and held-out final modes.

A five-iteration run generated a skill plus an MCP exposing objective
inspection, config validation, dev/final evaluation, score-component analysis,
deterministic hybrid search, one-shot autotune/apply/record, neighbor ranking,
rollback-safe candidate application, history comparison, and result recording.

| Arm | Attempts | Cumulative attempt time | Final held-out score |
| --- | --- | --- | --- |
| Plain | 5 (all timed out) | 600.08 s | 1,008,322 |
| **Harnessed** | **1 (timed out)** | **120.02 s** | **130,223** |

**87.09% lower** held-out `best_cycles`, reached with about **5× less**
cumulative attempt time. The generated harness was activated in a fresh replay
workspace, used by real Codex through MCP, updated `kernel_config.json`, passed
the final verifier, and beat the no-harness baseline.

---

## C flash attention (CPU)

Optimizes `kernel.c` for the lowest `best_cycles`. The benchmark is hardened
against benchmark-only tricks: it rejects source that patches Python timing,
validates every timed repeat against varied inputs, and includes held-out guard
cases.

A two-iteration deep-harness run built a skill plus an MCP exposing
`run_benchmark`, `numerical_check`, `assembly_summary`, `benchmark_variant`,
`score_delta_profile`, `sweep_kernel_experiments`, and `rank_experiments`, with
a self-test covering framing, every tool, numerical cases, dev/final
entrypoints, rollback, and error paths.

| Arm | Final held-out score (cycles) |
| --- | --- |
| Plain | 189,498 |
| **Harnessed** | **169,005** |

Both arms completed two timed-out 300 s attempts; the harnessed arm reached a
better verified score in the same budget.

---

## C++ stencil

Optimizes a five-point stencil in `kernel.cpp`. The benchmark compiles with the
system C++ compiler, checks correctness, reports cycles, and can emit assembly
diagnostics. It is hardened against patching `time.perf_counter_ns` from the
shared library.

A five-iteration run grew the generated MCP from **10 to 15 active tools** —
numerical tests, shape-alias tests, benchmark repeats, assembly diagnostics,
rollback-safe source trials, NEON row sweeps, dispatch sweeps, microvariant
sweeps, history, and ranking.

| Arm | Final score (cycles) |
| --- | --- |
| Plain | 43,200 |
| **Harnessed** | **34,934** |

**19.13% lower** held-out `best_cycles` in the same attempt budget.

---

## H100 Triton RMSNorm

The only bundled GPU task. Codex runs locally with the `exec` runner while all
objective scoring runs on a real NVIDIA H100 80GB HBM3 host over SSH.

A two-iteration run improved the independently-verified H100 final score from
**150.016 µs to a best checkpoint of 103.328 µs**, generated a skill plus MCP
server, repaired a qualification failure, and confirmed **10 generated MCP
calls** in the next fresh Codex session.

A longer four-iteration follow-up started from that generated bundle and
improved the verified score from **142.848 µs to 99.744 µs**, expanding the
committed H100 MCP to **17 active tools**: remote health checks, exact and
approximate SiLU source sweeps, joint source-plus-launch search, repeated
scoring, and next-experiment ranking.

See [H100 Triton RMSNorm](experiments/h100-triton-rmsnorm-2026-05-26.md) and
[the long run](experiments/h100-triton-rmsnorm-long-2026-05-27.md).

---

## Paged attention decode

The original end-to-end proof. Starting from a correct but slow paged-attention
decode kernel, a real Codex run generated a skill plus MCP server, then replayed
the original task with and without those artifacts:

| Arm | `best_ms` | Attempt duration |
| --- | --- | --- |
| Plain | 1.6759 | 144.6 s |
| **Harnessed** | **1.3588** | **112.7 s** |

Generated artifacts: `.harnessgym/skills/paged_attention_opt/SKILL.md` and
`.harnessgym/mcp/paged_attention_harness/`. The harnessed replay activated the
skill into `.agents/skills` and the MCP server into `.codex/config.toml`.

---

## Reproducing

Every result above is a `harnessgym run` followed by a `harnessgym compare`,
with the exact commands recorded in the
[experiment notes](experiments/index.md) and the project README. The general
shape is:

```bash
# 1. Generate a harness on a training workspace.
harnessgym run --task task.md --workspace . --iterations 5 \
  --runner exec --optimization-mode --score-key best_cycles --stop-score 1 \
  --post-attempt-command "python3 benchmark.py --json --mode final" \
  --post-attempt-score-key best_cycles

# 2. Replay plain vs harnessed from a clean template, equal budgets.
harnessgym compare \
  --workspace-template examples/<task> --task task.md \
  --artifact-source <run-workspace-or-.harnessgym> \
  --output-dir tmp/<task>_compare --trials 1 --iterations 5 \
  --attempt-timeout 5m --runner exec \
  --score-key best_cycles --stop-score 1 \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles
```

[**Replay & Compare →**](concepts/compare.md){ .md-button }
[**Examples →**](examples/index.md){ .md-button }
