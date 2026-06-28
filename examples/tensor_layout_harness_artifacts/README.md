# Tensor Layout Harness Artifacts

This directory contains the reusable `.harnessgym/` artifacts generated and
qualified during the tensor-layout optimization experiment. It is intentionally
separate from `examples/tensor_layout_pipeline_task/` so plain replay trials do
not see the harness bundle unless it is explicitly passed as `--artifact-source`.

Use it as the artifact source for replay validation:

```bash
harnessgym compare \
  --workspace-template examples/tensor_layout_pipeline_task \
  --task task.md \
  --artifact-source examples/tensor_layout_harness_artifacts/.harnessgym \
  --output-dir tmp/tensor_layout_claude_compare_final \
  --trials 1 \
  --iterations 1 \
  --attempt-timeout 5m \
  --runner claude \
  --score-key best_cycles \
  --stop-score 1 \
  --task-state continue \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles \
  --post-timeout 2m \
  --require-harness-tool-use \
  --overwrite
```

Or run the same validation with a Claude availability preflight:

```bash
examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```

The helper runs `check_claude_compare_report.py` after the compare. The checker
requires valid plain and harnessed trials, a valid JSON post-command score, and
at least one active generated MCP server with all 15 tensor-plan tools in the
harnessed trial. By default the helper also sets
`REQUIRE_HARNESS_TOOL_USE=1`, which requires the harnessed Claude attempt to
record at least one generated MCP `tools/call` in `.harnessgym/mcp_calls.jsonl`.

Optional environment overrides:

```bash
CLAUDE_BIN=claude ATTEMPT_TIMEOUT=10m TRIALS=2 OUTPUT_DIR=tmp/tensor_layout_claude_compare_final \
  examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```

Set `REQUIRE_HARNESS_TOOL_USE=0` only when you are debugging activation without
requiring the model to call the generated MCP tools.

To wait for a Claude quota reset instead of failing immediately at preflight:

```bash
WAIT_FOR_CLAUDE=1 PREFLIGHT_RETRY_SECONDS=300 PREFLIGHT_MAX_WAIT_SECONDS=14400 \
  examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```
