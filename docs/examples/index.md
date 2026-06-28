# Examples

The repository ships a set of self-contained task workspaces under
`examples/`. Each has a `task.md`, a benchmark/verifier, and (for the kernel
tasks) fast dev and held-out final modes. They range from a pure-Python warm-up
to a real GPU task, and two of them ship a committed, pre-generated harness
bundle you can replay against immediately.

| Example | Kind | Objective |
| --- | --- | --- |
| `numerical_debug_task` | pure Python | fix `normalized_dot` to return cosine similarity |
| `paged_attention_optimization_task` | Python kernel | reduce `best_ms` of a paged-attention decode kernel |
| `cpu_attention_autotune_task` | config autotune | reduce `best_cycles` editing `kernel_config.json` only |
| `c_flash_attention_optimization_task` | C kernel | reduce `best_cycles` in `kernel.c` |
| `cpp_stencil_kernel_task` | C++ kernel | reduce `best_cycles` in a five-point stencil |
| `cpu_moe_kernel_task` | C kernel | reduce `best_cycles` of a top-2 MoE kernel |
| `tensor_layout_pipeline_task` | layout/DMA | reduce `best_cycles` editing `kernel_plan.json` |
| `triton_rmsnorm_h100_task` | GPU (H100) | reduce `best_us` of a Triton RMSNorm, scored over SSH |

Two committed harness bundles let you skip generation and replay directly:
`examples/tensor_layout_harness_artifacts/` and
`examples/triton_rmsnorm_h100_harness_artifacts/`.

---

## Start here: the offline demo

`numerical_debug_task` is pure Python and runs with the `fake` runner — no agent
account, no network:

```bash
harnessgym run \
  --task examples/numerical_debug_task/task.md \
  --workspace examples/numerical_debug_task \
  --iterations 2 --attempt-timeout 10s --build-timeout 10s \
  --runner fake
```

---

## Tensor layout pipeline

The recommended performance-proof task. Optimizes `kernel_plan.json` — tiling,
layouts, vector width, DMA staging, prefetch, split-K, scheduling, swizzling,
epilogue fusion — with trace JSON exposing per-case component breakdowns.

```bash
harnessgym run \
  --task examples/tensor_layout_pipeline_task/task.md \
  --workspace examples/tensor_layout_pipeline_task \
  --iterations 5 --attempt-timeout 5m --reflection-timeout 3m --build-timeout 6m \
  --runner exec --stop-score 1 --score-key best_cycles \
  --task-state continue --harness-depth deep
```

A committed bundle is kept *outside* the workspace template so plain trials
don't see it, ready for a replay comparison:

```bash
harnessgym compare \
  --workspace-template examples/tensor_layout_pipeline_task --task task.md \
  --artifact-source examples/tensor_layout_harness_artifacts/.harnessgym \
  --output-dir tmp/tensor_layout_compare \
  --trials 1 --iterations 1 --attempt-timeout 5m --runner claude \
  --score-key best_cycles --stop-score 1 --task-state continue \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles --post-timeout 2m \
  --require-harness-tool-use --overwrite
```

The bundle also ships `run_claude_compare.sh`, a preflight + audit wrapper that
defaults to `REQUIRE_HARNESS_TOOL_USE=1`.

---

## CPU kernel tasks

These optimize real compiled kernels and have fast dev / held-out final modes
plus optional assembly diagnostics:

=== "CPU attention autotune"

    ```bash
    harnessgym run \
      --task examples/cpu_attention_autotune_task/task.md \
      --workspace examples/cpu_attention_autotune_task \
      --iterations 5 --attempt-timeout 2m --reflection-timeout 2m --build-timeout 4m \
      --runner exec --stop-score 1 --score-key best_cycles \
      --task-state continue --harness-depth deep
    ```

=== "C flash attention"

    ```bash
    harnessgym run \
      --task examples/c_flash_attention_optimization_task/task.md \
      --workspace examples/c_flash_attention_optimization_task \
      --iterations 5 --attempt-timeout 3m --build-timeout 3m \
      --runner exec --stop-score 1 --score-key best_cycles --task-state continue
    ```

=== "C++ stencil"

    ```bash
    harnessgym run \
      --task examples/cpp_stencil_kernel_task/task.md \
      --workspace examples/cpp_stencil_kernel_task \
      --iterations 5 --attempt-timeout 5m --reflection-timeout 3m --build-timeout 6m \
      --runner exec --optimization-mode --score-key best_cycles --stop-score 1 \
      --post-attempt-command "python3 benchmark.py --json --mode final" \
      --post-attempt-score-key best_cycles
    ```

=== "CPU top-2 MoE"

    ```bash
    harnessgym run \
      --task examples/cpu_moe_kernel_task/task.md \
      --workspace examples/cpu_moe_kernel_task \
      --iterations 5 --attempt-timeout 8m --reflection-timeout 3m --build-timeout 8m \
      --runner exec --optimization-mode \
      --post-attempt-command "python3 verifier.py --mode final --json" \
      --post-attempt-score-key best_cycles --post-attempt-timeout 2m \
      --stop-score 850000 --score-key best_cycles \
      --task-state continue --harness-depth deep
    ```

The kernel benchmarks are hardened against benchmark-only tricks — they reject
source that patches Python timing or dynamic symbol lookup, validate every timed
repeat against varied inputs, and include held-out guard cases.

---

## H100 Triton RMSNorm (real GPU)

The only GPU task. Codex runs locally; all objective scoring runs on a real
NVIDIA H100 80GB host over SSH via `remote_h100.py`.

```bash
rm -rf tmp/h100_triton_real && mkdir -p tmp/h100_triton_real
cp -R examples/triton_rmsnorm_h100_task/. tmp/h100_triton_real/

HARNESSGYM_GPU_HOST=<user@h100-host> \
HARNESSGYM_GPU_PORT=<ssh-port> \
HARNESSGYM_GPU_KEY=~/.ssh/id_ed25519 \
PYTHONPATH=src \
python3 -m harnessgym.cli run \
  --task tmp/h100_triton_real/task.md \
  --workspace tmp/h100_triton_real \
  --iterations 2 --attempt-timeout 5m --reflection-timeout 3m --build-timeout 5m \
  --post-attempt-command 'python3 remote_h100.py --workspace h100_triton_real_post -- python3 verifier.py --json --mode final --warmup 10 --repeats 20' \
  --post-attempt-score-key best_us --post-attempt-timeout 3m \
  --score-key best_us --stop-score 90 --optimization-mode --runner exec
```

The recorded outcomes for these examples are on the [Results](../results.md)
page; the full run notes are in [Experiments](../experiments/index.md).
