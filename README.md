<p align="center">
  <img src="https://raw.githubusercontent.com/patrick-toulme/harnessgym/main/docs/assets/harnessgym.png" alt="HarnessGym logo" width="520">
</p>

# HarnessGym

📖 **Documentation: [harnessgym.com](https://harnessgym.com)** — the docs are
built from `docs/` with MkDocs Material and deployed on every push to `main`.

HarnessGym is an open-source framework for iterative agent harness improvement. It runs a coding agent on a hard task, reflects in the same session on which reusable harness artifact would have helped most, builds that single artifact under `.harnessgym/`, and starts the next iteration in a fresh session with the accumulated registry context.

The package is alpha software for developers evaluating agent workflows. The core package has no third-party runtime dependencies; runner backends shell out to the agent CLI you choose. Codex and Claude Code are supported, and the deterministic `fake` runner works offline for smoke tests and demos.

## Install

```bash
python -m pip install harnessgym
harnessgym --help
```

For a source checkout with tests and the bundled examples:

```bash
git clone https://github.com/patrick-toulme/harnessgym.git
cd harnessgym
python -m pip install -e ".[dev]"
python -m pytest
```

Runner prerequisites:

- `--runner exec` uses the `codex` CLI, configurable with `--codex-bin`.
- `--runner claude` uses the `claude` CLI, configurable with `--claude-bin`.
- `--runner fake` is deterministic and does not require an agent account.
- Some examples below require additional local tooling such as NumPy, a C/C++ compiler, PyTorch, Triton, or access to a remote GPU.

## Quickstart

Create or choose a task file in an existing workspace, then run HarnessGym against that workspace:

```bash
harnessgym run \
  --task task.md \
  --workspace . \
  --iterations 3 \
  --attempt-timeout 45m \
  --build-timeout 20m \
  --runner exec
```

For an offline smoke test from a source checkout, use the bundled numerical debugging demo:

```bash
harnessgym run \
  --task examples/numerical_debug_task/task.md \
  --workspace examples/numerical_debug_task \
  --iterations 2 \
  --attempt-timeout 10s \
  --build-timeout 10s \
  --runner fake
```

## CLI

Optimization tasks can stop on an objective score instead of only `status: solved`:

```bash
harnessgym run \
  --task task.md \
  --workspace . \
  --iterations 2 \
  --attempt-timeout 5m \
  --build-timeout 5m \
  --runner exec \
  --stop-score 2.0 \
  --score-key best_ms
```

For time-boxed optimization tasks, use post-attempt scoring so HarnessGym independently verifies the workspace after every attempt, even if the runner process is killed before it updates `result.json`:

```bash
harnessgym run \
  --task task.md \
  --workspace . \
  --iterations 5 \
  --attempt-timeout 5m \
  --runner exec \
  --optimization-mode \
  --score-key best_cycles \
  --stop-score 1 \
  --post-attempt-command "python3 benchmark.py --json --mode final" \
  --post-attempt-score-key best_cycles
```

In optimization mode, `summary.json` records the baseline score, best score, relative improvement/reduction, and per-iteration harness usage. This avoids losing a real improvement when the attempt times out after mutating the workspace but before writing its final result JSON. By default, HarnessGym also checkpoints the best independently scored task workspace and restores it after the run, while leaving `.harnessgym`, `.codex`, `.agents`, and `.claude` activation state intact. Use `--no-restore-best` if you want to inspect the final attempted state instead.

Use `--attempt-timeouts 5m,8m` for per-iteration attempt budgets. Use `--build-after-solve` when you want a solved run to still reflect and build reusable harness artifacts for later replay experiments.

HarnessGym defaults to `--harness-depth deep`, which steers reflection/build toward capability-building harnesses instead of lightweight notes: multi-tool MCP servers with executable inspection and automation, such as compiler/IR analysis, assembly signals, benchmark history, variant sweeps, trace/layout tools, rollback-safe experiment ranking, and comprehensive self-tests. Use `--harness-depth standard` for smaller focused artifacts.

By default, generated artifacts are qualified before they are promoted to the next fresh attempt. After each build phase HarnessGym copies the pre-run task workspace into `.harnessgym/runs/<run_id>/qualification/`, copies in only the reusable `.harnessgym/` artifact bundle, activates generated skills/MCPs there, and runs MCP self-tests. If qualification fails, HarnessGym sends the exact failure report back to the same runner session and allows `--artifact-repair-attempts` repair builds, default `1`. Artifacts that still fail are marked quarantined in `registry.json`, hidden from future attempt prompts and activation, and preserved for repair evidence. Use `--no-qualify-artifacts` only when debugging the framework itself.

Task state has two modes:

```bash
# Compound task progress across iterations. This is the default.
harnessgym run --task task.md --workspace . --task-state continue

# Restore task files before each new iteration, while keeping accumulated .harnessgym artifacts.
harnessgym run --task task.md --workspace . --task-state reset
```

Use `continue` when you want the agent to keep improving the same working tree. Use `reset` when you want each fresh session to face the original task with only the generated harness artifacts carried forward.

Claude Code can be used instead of Codex:

```bash
harnessgym run \
  --task task.md \
  --workspace . \
  --iterations 3 \
  --attempt-timeout 45m \
  --build-timeout 20m \
  --runner claude
```

Useful Claude-specific flags:

- `--claude-bin`: path to the Claude Code executable, default `claude`.
- `--claude-model`: optional model alias or full model name, such as `sonnet` or `opus`.
- `--claude-permission-mode`: defaults to `bypassPermissions` for autonomous HarnessGym runs.
- `--claude-max-budget-usd`: optional per-phase spend cap for Claude Code print mode.
- `--claude-extra-arg`: repeatable escape hatch for newer Claude Code CLI flags.

Inline task text is also supported:

```bash
harnessgym run \
  --task-text "Fix the failing tests and verify them." \
  --workspace . \
  --iterations 2
```

Replay A/B comparisons are supported with `compare`. This copies a clean workspace template for each trial, runs attempt-only replays with and without a generated `.harnessgym/` artifact bundle, optionally runs a final JSON benchmark command, and writes `compare_report.json`. Failed post commands are recorded as invalid worst-case outcomes with `post_valid: false`, `post_invalid_reason`, and invalid counts in the summary instead of being silently treated as missing scores. Harnessed trials also require at least one generated MCP tool to activate by default; otherwise the trial is marked invalid for comparison so a broken artifact bundle cannot be mistaken for a harness win. Use `--no-require-active-harness` for smoke tests that intentionally copy only docs/scripts.

HarnessGym records concrete generated MCP `tools/call` telemetry to `.harnessgym/mcp_calls.jsonl` and surfaces call counts, called tool names, and compact samples in `compare_report.json`. Claude records this through its MCP bridge. Codex records this through the Content-Length MCP telemetry proxy used by its generated MCP config and by the workspace-local helper `python3 .harnessgym/runtime/mcp_call.py --server <name> --tool <tool> --arguments '<json-object>'`. Use `--require-harness-tool-use` when you want a harnessed trial to count only if the agent actually called at least one generated MCP tool, not merely activated it.

```bash
harnessgym compare \
  --workspace-template examples/c_flash_attention_optimization_task \
  --task task.md \
  --artifact-source tmp/c_flash_attention_5iter_real_20260517165223/.harnessgym \
  --output-dir tmp/c_flash_attention_compare \
  --trials 2 \
  --attempt-timeout 5m \
  --runner exec \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles
```

The repo also includes a committed tensor-layout harness bundle generated by a
previous qualified run. It is kept outside the workspace template so plain
trials do not see the harness artifacts:

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

The same replay is available as a helper script that first checks whether
Claude Code can make model calls, then audits the resulting compare report for
valid plain/harnessed trials, valid post scores, and the expected active
tensor-plan MCP tool count. The helper defaults to `REQUIRE_HARNESS_TOOL_USE=1`,
so a harnessed Claude replay must record at least one generated MCP tool call:

```bash
examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```

For quota-window validation, set `WAIT_FOR_CLAUDE=1` to retry the preflight for
up to `PREFLIGHT_MAX_WAIT_SECONDS` before running the compare.

The repo also includes a real H100-backed Triton task and the generated
HarnessGym artifact bundle from a verified run:

```bash
rm -rf tmp/h100_triton_real
mkdir -p tmp/h100_triton_real
cp -R examples/triton_rmsnorm_h100_task/. tmp/h100_triton_real/

HARNESSGYM_GPU_HOST=<user@h100-host> \
HARNESSGYM_GPU_PORT=<ssh-port> \
HARNESSGYM_GPU_KEY=~/.ssh/id_ed25519 \
PYTHONPATH=src \
python3 -m harnessgym.cli run \
  --task tmp/h100_triton_real/task.md \
  --workspace tmp/h100_triton_real \
  --iterations 2 \
  --attempt-timeout 5m \
  --reflection-timeout 3m \
  --build-timeout 5m \
  --post-attempt-command 'python3 remote_h100.py --workspace h100_triton_real_post -- python3 verifier.py --json --mode final --warmup 10 --repeats 20' \
  --post-attempt-score-key best_us \
  --post-attempt-timeout 3m \
  --score-key best_us \
  --stop-score 90 \
  --optimization-mode \
  --runner exec
```

The recorded experiment improved the H100 final score from `150.016 us` to a
best checkpoint of `103.328 us`, generated a skill plus MCP server, repaired a
qualification failure, and confirmed 10 generated MCP calls in the next fresh
Codex session. See `docs/experiments/h100-triton-rmsnorm-2026-05-26.md` and
`examples/triton_rmsnorm_h100_harness_artifacts/`.

A longer four-iteration follow-up started from that generated harness bundle and
improved the independently verified H100 score from `142.848 us` to
`99.744 us`. That run expanded the committed H100 MCP to 17 active tools,
including remote health checks, exact/approximate SiLU source sweeps, joint
source-plus-launch search, repeated scoring, and next-experiment ranking. See
`docs/experiments/h100-triton-rmsnorm-long-2026-05-27.md`.

## Workflow

Each iteration:

1. Starts a fresh runner session for the attempt phase.
2. Attempts the primary task until solved, blocked, failed, or timed out.
3. Reflects in the same session on the highest-leverage missing skill, MCP server, verifier, fixture, script, docs, or tool.
4. Builds the single selected improvement in the same session.
5. Qualifies generated artifacts in a clean replay workspace, repairs failed harness artifacts in the same session when possible, and quarantines anything still failing.
6. Stores reusable, promoted artifacts under `.harnessgym/` and records them in `.harnessgym/registry.json`.
7. Starts the next iteration with a fresh session and the accumulated registry context.

HarnessGym stops early when an iteration writes `result.json` with `status: "solved"` and `verified: true` or a passed verification object.

With `--task-state reset`, HarnessGym snapshots the non-`.harnessgym` workspace before the run and restores it before iterations 2..N. `.harnessgym` remains intact, so skills, MCP servers, tools, fixtures, and registry entries accumulate.

## Artifacts

Run-specific logs are stored under:

```text
.harnessgym/runs/<run_id>/iterations/<n>/
```

Each iteration directory contains:

- `result.json`
- `<phase>.prompt.txt`
- `<phase>.stdout.txt`
- `<phase>.stderr.txt`
- `<phase>.transcript.txt`

Reusable artifacts live under:

```text
.harnessgym/skills/
.harnessgym/mcp/
.harnessgym/tools/
.harnessgym/verifiers/
.harnessgym/fixtures/
.harnessgym/tests/
.harnessgym/docs/
.harnessgym/scripts/
```

The registry file `.harnessgym/registry.json` is synchronized from those directories after each build phase.

Generated tooling is expected to carry its own tests. For MCP servers, manifests can declare `self_test`; `true` means HarnessGym will run the server command with `--self-test`, while a command string/list/object can point at a separate test runner. For numerical, kernel, compiler, or benchmark tooling, the generated tests should include known toy cases, tolerance checks, fixed-seed randomized cases, and fast/dev plus final/held-out modes when applicable.

Failed artifacts remain on disk but are quarantined in `registry.json` with qualification metadata. Quarantined artifacts are not advertised in attempt prompts and are not injected into runner MCP config; the qualification report path gives the next repair build the concrete failure evidence.

## Agent Activation

HarnessGym keeps generated artifacts repo-local under `.harnessgym/`, then activates runner-native pieces before each fresh attempt:

- Skills from `.harnessgym/skills/<name>/SKILL.md` are symlinked into `.agents/skills/<name>` for Codex and `.claude/skills/<name>` for Claude Code.
- MCP manifests from `.harnessgym/mcp/**/{mcp.json,server.json,harnessgym-mcp.json}` are written into project-local `.codex/config.toml`.
- MCP servers are smoke-checked during activation with `initialize` and `tools/list`; failures are recorded as warnings instead of silently advertising a broken server.
- MCP servers must expose a non-empty tool inventory and pass a manifest `self_test` during activation before they are injected into a fresh runner session. `self_test: true` means HarnessGym runs the server command with `--self-test`; a command string/list/object can point at a separate test runner.
- The `exec` runner converts passed activation records into `codex exec -c mcp_servers...` overrides, because `codex exec` loads the user config by default and should not require mutating global Codex config for repo-local generated MCPs. Generated MCP servers are launched through `harnessgym.mcp_telemetry_proxy`, which preserves Content-Length MCP framing while logging tool calls. Activation also writes `.harnessgym/runtime/mcp_call.py`; Codex exec workers should use this helper when native MCP callables are not visible in the session instead of writing ad hoc JSON-RPC clients.
- The `claude` runner writes passed activation records into `.harnessgym/claude_mcp_config.json`, launches Claude Code with `--strict-mcp-config --mcp-config <that file>`, and grants generated MCP tools with Claude's server-level permission token `--allowedTools=mcp__<server>`. Because Claude Code 2.1.133 sends newline-delimited MCP JSON over stdio while HarnessGym-generated Codex MCP servers use Content-Length framing, the runner wraps each generated server with `harnessgym.claude_mcp_bridge`. The bridge response timeout follows each manifest's MCP tool timeout.
- The Codex MCP telemetry proxy and Claude MCP bridge both log each generated MCP `tools/call` as compact JSONL under `.harnessgym/mcp_calls.jsonl`, including server name, tool name, argument-key summary, duration, status, and result size. Compare reports include `mcp_telemetry`, `mcp_call_count`, and `mcp_called_tools`.
- MCP servers with failed smoke checks, missing/failing self-tests, or empty tool inventories are not injected into runner attempts.
- Generated MCP servers are expected to use Content-Length framed stdio JSON-RPC. Newline-delimited JSON helpers may be useful as local scripts, but they will fail Codex and Claude MCP activation.
- Activation details, quality-gate status, and active tool counts are captured in `.harnessgym/activation.json`, each iteration's `activation.json`, and post-build activation snapshots. HarnessGym also infers harness usage from attempt stdout/stderr/transcripts so timeout runs can still report generated tool use.
- Post-build fresh-workspace qualification reports are captured under each iteration's `qualification/attempt_<n>/qualification.json`, and repair build logs are captured under `repair_<n>/`.

This means the next fresh Codex or Claude Code session can see generated skills and MCP servers while the source of truth remains under `.harnessgym/`.

## Runners

- `exec`: MVP backend. Uses `codex exec` for attempts and `codex exec resume <session_id>` for reflection/build phases when Codex exposes a session id. It uses ordinary autonomous prompts; it does not assume `codex exec "/goal ..."` sets a goal. Timeout handling launches Codex in a process group and terminates the whole group so spawned benchmark/tool children cannot keep pipes open indefinitely.
- `claude`: Claude Code backend. Uses `claude -p --output-format json` for attempts and `claude -p --resume <session_id>` for reflection/build phases. It parses Claude's JSON `session_id`, captures stdout/stderr/transcripts, enforces process-group timeouts, activates generated skills under `.claude/skills`, and passes qualified MCP servers through a repo-local Claude MCP config plus the stdio framing bridge.
- `tui-goal`: experimental PTY backend. It launches interactive `codex`, sends a real `/goal` command for the attempt phase, and sends reflection/build prompts to the same process. Completion is inferred from `result.json`, so the `exec` runner is the recommended MVP path.
- `fake`: deterministic offline runner for tests and demos. It simulates a failed first attempt, same-session reflection, artifact creation, and a fresh second attempt that uses generated artifact context.

## Demo

The repo includes a pure-Python numerical debugging task:

```bash
harnessgym run \
  --task examples/numerical_debug_task/task.md \
  --workspace examples/numerical_debug_task \
  --iterations 2 \
  --attempt-timeout 2m \
  --build-timeout 2m \
  --runner exec
```

Offline deterministic demo:

```bash
harnessgym run \
  --task examples/numerical_debug_task/task.md \
  --workspace examples/numerical_debug_task \
  --iterations 2 \
  --attempt-timeout 10s \
  --build-timeout 10s \
  --runner fake
```

The fake demo intentionally blocks the first attempt, creates `.harnessgym/tools/harnessgym_fake_probe.py`, updates the registry, starts a fresh second attempt with that registry context, applies the known demo fix, runs `python verifier.py`, and records the verified result.

Harder optimization demo:

```bash
harnessgym run \
  --task examples/paged_attention_optimization_task/task.md \
  --workspace examples/paged_attention_optimization_task \
  --iterations 2 \
  --attempt-timeout 5m \
  --build-timeout 5m \
  --runner exec \
  --stop-score 2.0 \
  --score-key best_ms
```

This benchmark starts from a correct but slow paged-attention decode kernel. The objective is to reduce `best_ms` from `python3 benchmark.py --json`; correctness remains mandatory.

The real validation run used a training workspace to generate a Codex skill and MCP server, then replayed the original task with and without those artifacts:

- Harnessed replay: `best_ms=1.3588`, attempt duration `112.7s`.
- No-artifact replay: `best_ms=1.6759`, attempt duration `144.6s`.
- Generated artifacts included `.harnessgym/skills/paged_attention_opt/SKILL.md` and `.harnessgym/mcp/paged_attention_harness/`.
- The harnessed replay activated the skill into `.agents/skills` and MCP server into `.codex/config.toml`.

These numbers are from one real Codex run on this machine, not a statistically powered benchmark. They are useful as an end-to-end proof that generated harness artifacts can reduce solve time and improve the reached score; repeat runs may vary.

CPU C Flash Attention optimization demo:

```bash
harnessgym run \
  --task examples/c_flash_attention_optimization_task/task.md \
  --workspace examples/c_flash_attention_optimization_task \
  --iterations 5 \
  --attempt-timeout 3m \
  --build-timeout 3m \
  --runner exec \
  --stop-score 1 \
  --score-key best_cycles \
  --task-state continue
```

This demo optimizes `kernel.c` for the lowest `best_cycles` from `python3 benchmark.py --json`. The benchmark uses randomized Q/K/V inputs, verifies every timed output, and avoids warming the exact measured case before timing, preventing repeated-input memoization from counting as a real kernel speedup.

The benchmark has two modes:

```bash
python3 benchmark.py --json --mode dev
python3 benchmark.py --json --mode final
```

`dev` is the fast feedback case used in the task prompt. `final` aggregates held-out shape/seed cases and is better for replay comparisons.

The real five-iteration validation run captured:

- Starting example score before Codex: about `816k` cycles.
- Iteration scores: `755231`, `189648`, `188915`, `124112`, `139087` best observed cycles.
- Final authoritative benchmark after the run: `115862` cycles, correctness passed.
- Generated artifacts included `.harnessgym/skills/c-flash-attention-harness/SKILL.md` and `.harnessgym/mcp/c-flash-attention-harness/`.

The final status is expected to be `incomplete` in this run because `--stop-score 1` is intentionally unreachable; it forces all five iterations to execute.

After hardening the evaluator with final held-out cases, a refreshed real HarnessGym build produced:

- `.harnessgym/skills/c-flash-attention-optimizer/SKILL.md`
- `.harnessgym/mcp/c-flash-attention-harness/server.py`
- `.harnessgym/mcp/c-flash-attention-harness/harnessgym-mcp.json`

The refreshed MCP server activated cleanly in a fresh replay workspace and exposed `c_flash_benchmark`, `c_flash_compare_modes`, `c_flash_compile_case`, and `c_flash_score_ranges`.

A one-trial real compare with equal 300 second attempt budgets captured:

- Plain replay final held-out score: `222581` cycles.
- Harnessed replay final held-out score: `189356` cycles.
- Both attempts timed out at the 300 second cap, so this proves better reached score in the same model-time budget, not lower wall-clock completion time.
- Report path: `tmp/c_flash_attention_compare_refresh_1trial/compare_report.json`.

After switching deep harness mode to require comprehensive generated-tooling tests, another real two-iteration generation run built:

- `.harnessgym/skills/c-flash-attention-optimizer/SKILL.md`
- `.harnessgym/mcp/c_flash_attention/server.py`
- `.harnessgym/mcp/c_flash_attention/harnessgym-mcp.json`
- `.harnessgym/tests/test_c_flash_attention_mcp.py`

The generated MCP exposed `run_benchmark`, `numerical_check`, `assembly_summary`, `benchmark_variant`, `score_delta_profile`, `sweep_kernel_experiments`, and `rank_experiments`. Its self-test covered Content-Length JSON-RPC framing, every tool, numerical toy/random cases, dev/final benchmark entrypoints, rollback behavior, and invalid-tool/error paths. A fresh harnessed replay activated the MCP with no warnings and passed the generated self-test.

A one-trial real two-iteration compare with equal 600 second cumulative attempt budgets captured:

- Plain replay final held-out score: `189498` cycles.
- Harnessed replay final held-out score: `169005` cycles.
- Both arms completed two timed-out 300 second attempts; the harnessed arm achieved a better verified score with the same model-time budget.
- Report path: `tmp/c_flash_attention_compare_deep_tests_2turn_20260518/compare_report.json`.

CPU attention autotune harness demo:

```bash
harnessgym run \
  --task examples/cpu_attention_autotune_task/task.md \
  --workspace examples/cpu_attention_autotune_task \
  --iterations 5 \
  --attempt-timeout 2m \
  --reflection-timeout 2m \
  --build-timeout 4m \
  --runner exec \
  --stop-score 1 \
  --score-key best_cycles \
  --task-state continue \
  --harness-depth deep
```

This is a deterministic CPU custom-kernel autotuning proxy. The task edits only `kernel_config.json`, verifies correctness, and optimizes `best_cycles` across fast dev and held-out final modes.

A real five-iteration HarnessGym generation run built:

- `.harnessgym/skills/cpu_attention_autotune/SKILL.md`
- `.harnessgym/mcp/cpu_attention_autotune/server.py`
- `.harnessgym/mcp/cpu_attention_autotune/harnessgym-mcp.json`
- `.harnessgym/tests/test_cpu_attention_mcp.py`

The generated MCP exposed objective inspection, config validation, dev/final evaluation, score-component analysis, deterministic hybrid search, one-shot autotune/apply/record, neighbor ranking, rollback-safe candidate application, history comparison, and result recording. Its self-test covered Content-Length MCP framing, basic tools, fixed-seed randomized numerical checks, dev/final entrypoints, search schema, apply rollback, and score-component agreement with the benchmark.

The clean real compare command was:

```bash
harnessgym compare \
  --workspace-template examples/cpu_attention_autotune_task \
  --task task.md \
  --artifact-source tmp/cpu_attention_autotune_harness5_20260518/.harnessgym \
  --output-dir tmp/cpu_attention_autotune_compare_pg_20260518 \
  --trials 1 \
  --iterations 5 \
  --attempt-timeout 2m \
  --runner exec \
  --score-key best_cycles \
  --stop-score 1 \
  --task-state continue \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles
```

Observed result:

- Plain replay: five timed-out attempts, `600.08s` cumulative attempt time, final held-out score `1008322`.
- Harnessed replay: one timed-out attempt, `120.02s` cumulative attempt time, final held-out score `130223`.
- Score reduction: `87.09%` lower held-out `best_cycles`.
- Attempt-time reduction: about `5.0x` less cumulative attempt time.
- Report path: `tmp/cpu_attention_autotune_compare_pg_20260518/compare_report.json`.

This is still one trial, not a statistical claim. It does prove the generated harness was activated in a fresh replay workspace, used by real Codex through MCP, updated `kernel_config.json`, wrote `result.json`, passed the final verifier, and beat the no-harness baseline by more than 50% on the objective.

Tensor layout pipeline harness demo:

```bash
harnessgym run \
  --task examples/tensor_layout_pipeline_task/task.md \
  --workspace examples/tensor_layout_pipeline_task \
  --iterations 5 \
  --attempt-timeout 5m \
  --reflection-timeout 3m \
  --build-timeout 6m \
  --runner exec \
  --stop-score 1 \
  --score-key best_cycles \
  --task-state continue \
  --harness-depth deep
```

This example is intentionally more comprehensive than the CPU-autotune proxy. It optimizes `kernel_plan.json`, which controls tile sizes, tensor layouts, vector width, softmax strategy, accumulation, DMA stages, prefetch distance, descriptor burst size, split-K, scratchpad size, scheduling, swizzling, and epilogue fusion. The benchmark exposes fast dev cases, held-out final cases, correctness tolerance, and detailed trace JSON:

```bash
python3 benchmark.py --json --mode dev --trace trace.dev.json
python3 benchmark.py --json --mode final
python3 verifier.py
```

The trace includes descriptor counts, register pressure, scratchpad need, compute/memory cycles, bank-conflict penalties, DMA penalties, and layout/schedule synergies. This gives HarnessGym enough signal to build higher-value harnesses such as tensor-layout analyzers, DMA descriptor tools, trace diff tools, rollback-safe plan sweepers, and MCP self-tests. The starting final score is `33975173` cycles and is valid but deliberately poor.

This is the recommended next performance-proof task when evaluating whether HarnessGym beats vanilla Codex. It is intentionally less hand-solvable than the stencil task because progress depends on inspecting trace/layout/descriptor interactions and automating sweeps rather than applying one obvious kernel rewrite.

A real five-iteration run generated a tensor-layout skill and MCP server with focused search, off-spine search, candidate proof, deadline-aware apply, fast focused apply, history, final gating, and self-tests. The final workspace scored:

- Dev: `262005` cycles, down from `8348345`.
- Final: `1495982` cycles, down from `33975173`.
- Final reduction: `95.60%`.

The CLI run still reported `Solved and verified: False` because the attempt was killed after changing `kernel_plan.json` and before Codex completed its own result write. The post-attempt scoring mode above was added to close that gap for future runs.

CPU top-2 MoE kernel demo:

```bash
harnessgym run \
  --task examples/cpu_moe_kernel_task/task.md \
  --workspace examples/cpu_moe_kernel_task \
  --iterations 5 \
  --attempt-timeout 8m \
  --reflection-timeout 3m \
  --build-timeout 8m \
  --runner exec \
  --post-attempt-command "python3 verifier.py --mode final --json" \
  --post-attempt-score-key best_cycles \
  --post-attempt-timeout 2m \
  --optimization-mode \
  --stop-score 850000 \
  --score-key best_cycles \
  --task-state continue \
  --harness-depth deep
```

This example is a real compiled C kernel benchmark for CPU top-2 mixture-of-experts inference. It optimizes `moe_kernel.c` plus `kernel_config.json`; `benchmark.py` compiles a shared library, generates deterministic routed MoE cases, checks numerical correctness against a Python reference, and reports estimated cycle counts from best-of-repeat timings:

```bash
python3 benchmark.py --json --mode dev --trace trace.dev.json
python3 benchmark.py --json --mode dev --assembly moe_kernel.s
python3 benchmark.py --json --mode final
python3 verifier.py --mode final --json
```

The dev/final cases include uniform, Zipf, bursty, and adversarial routing distributions. The trace records expert bucket load, route skew, work estimates, and per-case cycle data. This is designed to push HarnessGym toward route-bucket analyzers, assembly/vectorization summaries, config/source variant sweepers, correctness fuzzers, and rollback-safe final-mode ranking.

Realistic C++ stencil-kernel demo:

```bash
harnessgym run \
  --task examples/cpp_stencil_kernel_task/task.md \
  --workspace examples/cpp_stencil_kernel_task \
  --iterations 5 \
  --attempt-timeout 5m \
  --reflection-timeout 3m \
  --build-timeout 6m \
  --runner exec \
  --optimization-mode \
  --score-key best_cycles \
  --stop-score 1 \
  --post-attempt-command "python3 benchmark.py --json --mode final" \
  --post-attempt-score-key best_cycles
```

This task optimizes a C++ five-point stencil in `kernel.cpp`. The benchmark compiles the kernel with the system C++ compiler, checks numerical correctness, reports estimated cycles, and can emit compiler assembly diagnostics:

```bash
python3 benchmark.py --json --mode dev --trace trace.dev.json --assembly .harnessgym_build/kernel.s
python3 benchmark.py --json --mode final
python3 verifier.py
```

The assembly summary includes instruction-line count, branch mentions, vector-register mentions, FMA mentions, and load/store mentions. It is intended to induce harnesses that inspect compiler artifacts, benchmark rollback-safe variants, compare dev/final regressions, and maintain benchmark history.

The C++ benchmark is hardened against benchmark-only tricks: it rejects source that references Python timing/Python C APIs or dynamic symbol lookup, validates every timed repeat against varied input buffers, and includes guard cases with alternate alpha/seed values. This was added after a real compare exposed that a model could otherwise patch `time.perf_counter_ns` from the shared library and report zero-cycle work.

After a generation run, compare a fresh plain attempt against a fresh harnessed attempt with:

```bash
harnessgym compare \
  --workspace-template examples/cpp_stencil_kernel_task \
  --task task.md \
  --artifact-source <run-workspace-or-harnessgym-dir> \
  --output-dir tmp/cpp_stencil_compare \
  --trials 1 \
  --iterations 5 \
  --attempt-timeout 5m \
  --runner exec \
  --score-key best_cycles \
  --stop-score 1 \
  --task-state continue \
  --post-command "python3 benchmark.py --json --mode final" \
  --post-score-key best_cycles \
  --overwrite
```

A real five-iteration HarnessGym generation run on this task produced:

- `.harnessgym/skills/cpp-stencil-optimizer/SKILL.md`
- `.harnessgym/mcp/cpp-stencil-harness/server.py`
- `.harnessgym/mcp/cpp-stencil-harness/harnessgym-mcp.json`
- `.harnessgym/tests/test_cpp_stencil_mcp.py`

The generated MCP grew from 10 to 15 active tools across iterations, including numerical tests, shape-alias tests, benchmark repeats, assembly diagnostics, rollback-safe source trials, NEON row sweeps, dispatch sweeps, microvariant sweeps, history, and ranking. Activation quality gates passed with MCP smoke plus self-test.

After hardening the benchmark, a one-iteration real clean compare with equal 300 second attempt budgets captured:

- Plain replay final score: `43200` cycles.
- Harnessed replay final score: `34934` cycles.
- Score reduction: `19.13%` lower held-out `best_cycles` in the same attempt budget.
- Report path: `tmp/cpp_stencil_compare_hardened_1iter_20260518/compare_report.json`.

This is one trial, not a statistical claim. Its main value is that it exercises the real HarnessGym path end to end: generated artifacts copied into a fresh workspace, skill/MCP activation, MCP self-test, active tool use by Codex, final post-scoring, and invalid post-score handling.

## Development

```bash
python -m pytest
```

TODO:

- Add richer statistical scoring for multi-run compare reports.
- Add richer session-id parsing if Codex or Claude Code CLI output changes.
- Harden the `tui-goal` runner around interactive completion signals.
