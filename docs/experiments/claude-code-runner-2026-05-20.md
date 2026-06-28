# Claude Code Runner Validation - 2026-05-20

This note records the real validation done while adding the HarnessGym
`claude` runner.

## Implementation Summary

The runner uses Claude Code headless mode:

```bash
claude -p --output-format json
```

Reflection and build phases resume the same Claude Code session with
`--resume <session_id>`. The runner parses Claude's JSON `session_id`, captures
stdout/stderr/transcripts, and terminates the process group on timeout.

Generated HarnessGym skills are activated into both `.agents/skills/` and
`.claude/skills/`.

Generated MCP servers are activated for Claude Code through
`.harnessgym/claude_mcp_config.json`. Claude Code 2.1.133 sends newline JSON
over stdio for local MCP servers, while the existing HarnessGym/Codex MCP
servers use Content-Length framing, so the runner wraps each generated MCP
server with `harnessgym.claude_mcp_bridge`.

Claude MCP permissions use server-level tokens:

```bash
--allowedTools=mcp__<server-name>
```

Wildcards such as `mcp__<server-name>__*` were tested and are not the correct
Claude Code permission shape.

## Validation Commands

Unit suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test*.py' -v
```

Direct Claude Code edit smoke:

```bash
claude -p --output-format json --permission-mode bypassPermissions \
  "Create result.json containing {\"status\": \"solved\", \"verified\": true}."
```

Tensor-layout compare attempt with Claude Code:

```bash
PYTHONPATH=src python3 -m harnessgym.cli compare \
  --workspace-template examples/tensor_layout_pipeline_task \
  --task task.md \
  --artifact-source examples/tensor_layout_harness_artifacts/.harnessgym \
  --output-dir tmp/tensor_layout_claude_compare_bridge_20260520T043955Z \
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
  --overwrite
```

The committed artifact source above contains only the reusable, qualified
tensor-layout harness artifacts. The original generating run remains in ignored
`tmp/` state, but replay validation should use the committed
`examples/tensor_layout_harness_artifacts/.harnessgym` bundle.

The same replay command is wrapped by:

```bash
examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```

That helper first runs a tiny Claude Code model-call preflight so quota failures
do not create another invalid compare directory. After a successful compare it
runs `check_claude_compare_report.py`, which requires valid plain and harnessed
trials, valid post-command scores, and all 15 generated tensor-plan MCP tools to
be active in the harnessed trial.

## Real Results

Claude Code version:

```text
2.1.133 (Claude Code)
```

The first real compare before the bridge exposed the compatibility issue:

- Output: `tmp/tensor_layout_claude_compare_20260520T041115Z/compare_report.json`
- Plain Claude: final `best_cycles=1,492,112`
- Harnessed Claude: final `best_cycles=33,975,173`
- HarnessGym activation showed 1 MCP and 15 tools, but Claude reported the
  server as still connecting.

After adding the bridge, the same copied generated MCP connected through Claude
Code's own MCP health check:

```bash
cp .harnessgym/claude_mcp_config.json .mcp.json
MCP_TIMEOUT=30000 claude mcp get tensor-plan-server
```

Result:

```text
Status: ✓ Connected
```

A direct Claude Code visibility smoke against the bridged config succeeded and
listed all 15 generated tools, including:

- `mcp__tensor-plan-server__apply_best_verified`
- `mcp__tensor-plan-server__benchmark_plan`
- `mcp__tensor-plan-server__bounded_exhaustive_search`
- `mcp__tensor-plan-server__run_objective`
- `mcp__tensor-plan-server__validate_plan`

The follow-up full compare after the bridge started, but the harnessed Claude
attempt failed immediately with a Claude Code 429 usage limit:

```text
You're out of extra usage · resets 5am (America/New_York)
```

The report is preserved at:

```text
tmp/tensor_layout_claude_compare_bridge_20260520T043955Z/compare_report.json
```

Because of that quota limit, this run proves Claude Code runner startup,
session parsing, timeout handling, generated skill activation, and bridged MCP
connectivity on the verified tensor-layout artifact bundle. It does not prove a
successful post-bridge Claude A/B performance win.

The quota failure also exposed a compare-reporting issue: a failed runner
attempt could still be followed by a valid post-command score from the unchanged
workspace. Compare now records `runner_validation`, marks failed runner attempts
as `comparison_valid: false`, and treats their post score as invalid for A/B
summary purposes.

## Fresh MCP Health Recheck

At 2026-05-20 00:51 EDT, Claude Code model calls were still blocked with the
same 429 reset message, so the real compare could not be rerun yet. A fresh
non-model health check was run instead:

```text
tmp/tensor_layout_claude_mcp_health_20260520T045154Z
```

The verified tensor-layout `.harnessgym` bundle was copied into that clean
workspace, activated through HarnessGym, and then checked with Claude Code's
MCP health command.

HarnessGym activation:

```json
{
  "status": "passed",
  "active_mcp_count": 1,
  "inactive_mcp_count": 0,
  "active_tool_count": 15,
  "warnings": []
}
```

Claude Code health command:

```bash
cp .harnessgym/claude_mcp_config.json .mcp.json
MCP_TIMEOUT=30000 claude mcp get tensor-plan-server
```

Result:

```text
Status: ✓ Connected
```

This fresh check confirms the generated tensor-layout MCP server can be
activated from a copied replay workspace through the HarnessGym Claude bridge.
The remaining unproven item is a quota-unblocked model-backed Claude attempt on
the same verified experiment.

## Live Quota-Failure Compare Recheck

At 2026-05-20 00:53 EDT, a short real `harnessgym compare --runner claude`
was run while Claude Code was still returning 429, specifically to verify the
current compare hardening against the live Claude failure path:

```text
tmp/tensor_layout_claude_quota_invalid_20260520T045346Z/compare_report.json
```

Result:

- Plain trial: `comparison_valid: false`, `post_score: null`.
- Harnessed trial: `comparison_valid: false`, `post_score: null`.
- Both invalid reasons: `runner attempt failed in iteration(s): 1`.
- Harnessed activation still reported 1 active MCP and 15 active tools before
  the Claude model call failed.

This confirms failed Claude runner attempts are no longer counted as valid A/B
scores. The model-backed tensor-layout validation still needs a quota-unblocked
Claude run.

## Quota-Unblocked Model-Backed Compare

At 2026-05-20 05:12 EDT, after the Claude Code quota reset, the committed
validation helper completed the real model-backed tensor-layout replay:

```bash
WAIT_FOR_CLAUDE=1 PREFLIGHT_RETRY_SECONDS=60 PREFLIGHT_MAX_WAIT_SECONDS=14400 \
  examples/tensor_layout_harness_artifacts/run_claude_compare.sh
```

The helper waited through the reset window, then reported:

```text
Claude Code preflight passed after 12419s.
Running HarnessGym tensor-layout Claude compare...
```

Output:

```text
tmp/tensor_layout_claude_wait_final_20260520T053409Z/compare_report.json
```

Compare summary:

- Plain Claude trial: `comparison_valid: true`, `post_valid: true`,
  `post_score=1,495,982`, attempt timed out at `300.170s`.
- Harnessed Claude trial: `comparison_valid: true`, `post_valid: true`,
  `post_score=1,495,982`, attempt timed out at `300.194s`.
- Harnessed activation: `active_mcp_count=1`, `active_tool_count=15`,
  `warnings=[]`.
- Harnessed transcript command included `--strict-mcp-config`, the generated
  `.harnessgym/claude_mcp_config.json`, and
  `--allowedTools=mcp__tensor-plan-server`.
- The generated `tensor-plan-server` MCP smoke check and self-test both passed.

Post-command verifier output for both trials:

```json
{
  "status": "passed",
  "best_cycles": 1495982,
  "max_abs": 0.000089,
  "tolerance": 0.00025
}
```

The committed report checker passed:

```bash
python3 examples/tensor_layout_harness_artifacts/check_claude_compare_report.py \
  tmp/tensor_layout_claude_wait_final_20260520T053409Z/compare_report.json \
  --min-active-mcp 1 \
  --min-active-tools 15
```

Result:

```json
{
  "errors": [],
  "harnessed_best_active_mcp_count": 1,
  "harnessed_best_active_tool_count": 15,
  "harnessed_best_post_score": 1495982.0,
  "plain_best_post_score": 1495982.0,
  "report": "tmp/tensor_layout_claude_wait_final_20260520T053409Z/compare_report.json",
  "status": "passed"
}
```

This validates the Claude Code runner end to end on the verified tensor-layout
experiment: model-backed Claude attempts launched, timeouts were enforced,
stdout/stderr/transcripts were captured, generated `.harnessgym` skills and
MCP artifacts were copied and activated, the Claude MCP bridge/config was used,
the objective verifier ran after both attempts, and the machine-readable compare
report passed the committed audit. This run did not show a harness performance
win: both attempts timed out and the final objective score was unchanged.
