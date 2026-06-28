# Experiments

These are preserved notes from real HarnessGym validation runs — the raw
engineering record behind the [Results](../results.md) page. Each captures the
exact commands, the generated artifacts, the qualification/repair events, and
the measured before/after numbers.

They are evidence the mechanism works end to end on one machine, not
statistically powered benchmarks. Where an attempt timed out, the notes say so;
where an artifact was quarantined or repaired, the notes record it.

| Experiment | What it validated |
| --- | --- |
| [Tensor Layout Qualification](tensor-layout-qualification.md) | fresh-workspace artifact qualification end to end |
| [CPU MoE Real Smoke](cpu-moe-real-smoke.md) | the compiled top-2 MoE kernel task path |
| [H100 Triton RMSNorm](h100-triton-rmsnorm.md) | real H100-over-SSH scoring, 150.0 → 103.3 µs |
| [H100 Triton RMSNorm (Long)](h100-triton-rmsnorm-long.md) | longer run from a generated bundle, 17 active tools, 142.8 → 99.7 µs |

## How to read these

Each note follows the same arc the loop does:

1. **Task** — what was being optimized and how it was scored.
2. **Generation** — the `harnessgym run` command and the artifacts it produced.
3. **Qualification** — whether artifacts passed the clean-room gate, and any
   repair/quarantine events.
4. **Replay** — the `harnessgym compare` (or post-attempt scoring) and the
   plain-vs-harnessed numbers.

If you want to reproduce one, the commands in each note are the same ones on the
[Examples](../examples/index.md) page, run against the corresponding workspace.
