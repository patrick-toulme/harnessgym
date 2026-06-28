from __future__ import annotations

import json
from pathlib import Path

from .config import RunConfig
from .models import IterationContext, Registry
from .registry import artifact_is_quarantined


RESULT_SHAPE = {
    "status": "solved | blocked | incomplete | tooling_built | failed",
    "verified": "boolean",
    "summary": "short description of what happened",
    "blockers": ["optional blockers"],
    "missing_tooling": ["optional tooling gaps"],
    "reflection": {
        "recommendations": ["one or two high-leverage additions"],
        "selected_improvement": {
            "kind": "skill | mcp | tool | verifier | fixture | docs | script | test",
            "name": "short name",
            "reason": "why this is the highest leverage next addition",
            "target_path": ".harnessgym/<area>/...",
        },
    },
    "verification": {
        "status": "passed | failed | not_run",
        "command": "optional command",
        "tooling_tests": [
            {
                "name": "optional generated tool or MCP test",
                "status": "passed | failed | not_run",
                "command": "repo-local deterministic command",
            }
        ],
    },
    "metrics": {"score": "numeric objective score if this is an optimization task"},
}


def render_artifact_context(registry: Registry, workspace: Path | None = None, preview_chars: int = 3000) -> str:
    visible_artifacts = [artifact for artifact in registry.artifacts if not artifact_is_quarantined(artifact)]
    quarantined_count = len(registry.artifacts) - len(visible_artifacts)
    if not visible_artifacts:
        lines = ["No generated HarnessGym artifacts are registered yet."]
    else:
        lines = ["Registered HarnessGym artifacts available to this fresh attempt:"]
    if quarantined_count:
        lines.append(
            f"{quarantined_count} quarantined artifact(s) are hidden from attempts; "
            "see .harnessgym/registry.json and qualification reports for repair details."
        )
    if workspace is not None:
        lines.extend(_activation_context_lines(workspace))
    for artifact in visible_artifacts:
        description = f" - {artifact.description}" if artifact.description else ""
        lines.append(f"- {artifact.kind}: {artifact.path}{description}")
        if workspace is not None and artifact.kind in {"docs", "skill", "verifier", "fixture", "script", "tool", "test", "mcp"}:
            artifact_path = workspace / artifact.path
            if artifact_path.exists() and artifact_path.is_file():
                text = artifact_path.read_text(encoding="utf-8", errors="replace")
                preview = text[:preview_chars]
                if len(text) > preview_chars:
                    preview += "\n...[truncated; open artifact for full content]..."
                lines.append(f"  Preview:\n{_indent(preview, '  ')}")
    return "\n".join(lines)


def _activation_context_lines(workspace: Path) -> list[str]:
    activation_path = workspace / ".harnessgym" / "activation.json"
    if not activation_path.exists():
        return []
    try:
        activation = json.loads(activation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ["Active HarnessGym activation context: unreadable activation.json"]
    lines = ["Active HarnessGym activation context:"]
    warnings = activation.get("warnings") if isinstance(activation, dict) else None
    if warnings:
        lines.append(f"- Activation warnings: {warnings}")
    quality_gate = activation.get("quality_gate") if isinstance(activation, dict) else None
    if isinstance(quality_gate, dict):
        lines.append(
            "- Harness quality gate: "
            f"{quality_gate.get('status', 'unknown')}; "
            f"active MCPs={quality_gate.get('active_mcp_count', 0)}; "
            f"active tools={quality_gate.get('active_tool_count', 0)}"
        )
    servers = activation.get("mcp_servers", []) if isinstance(activation, dict) else []
    for server in servers:
        if not isinstance(server, dict):
            continue
        tools = ", ".join(str(tool) for tool in server.get("enabled_tools") or server.get("smoke", {}).get("tools", []))
        smoke = server.get("smoke", {}).get("status", "unknown") if isinstance(server.get("smoke"), dict) else "unknown"
        self_test = (
            server.get("self_test", {}).get("status", "unknown")
            if isinstance(server.get("self_test"), dict)
            else "unknown"
        )
        lines.append(
            f"- MCP `{server.get('name', 'unknown')}` active={bool(server.get('active', True))}; "
            f"smoke={smoke}; self_test={self_test}; tools=[{tools}]"
        )
    helper = activation.get("mcp_call_helper") if isinstance(activation, dict) else None
    if isinstance(helper, dict) and helper.get("relative_path"):
        lines.append(
            "- MCP call helper: "
            f"`python3 {helper['relative_path']} --server <server> --tool <tool> --arguments '<json-object>'`"
        )
    skills = activation.get("skills", []) if isinstance(activation, dict) else []
    for skill in skills:
        if isinstance(skill, dict):
            active_paths = [
                path
                for path in [skill.get("codex_activated_path"), skill.get("claude_activated_path")]
                if path
            ]
            if not active_paths and skill.get("activated_path"):
                active_paths.append(str(skill.get("activated_path")))
            lines.append(f"- Skill active at {', '.join(active_paths)}: {skill.get('artifact_path')}")
    return lines


def _indent(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return str(path)


def _agent_name(config: RunConfig) -> str:
    return "Claude Code" if config.runner == "claude" else "Codex"


def render_attempt_prompt(config: RunConfig, context: IterationContext) -> str:
    agent_name = _agent_name(config)
    task_origin = (
        _workspace_relative(context.task_path, context.workspace)
        if context.task_path is not None
        else "inline task text"
    )
    score_target = ""
    if config.stop_score is not None:
        direction = "greater than or equal to" if config.higher_is_better else "less than or equal to"
        score_target = (
            f"\nOptimization stop target: record `{config.score_key}` and stop the attempt once it is "
            f"{direction} {config.stop_score}. Also mirror the same numeric value to `metrics.score`.\n"
        )
    return f"""You are running HarnessGym iteration {context.iteration} in workspace:
{context.workspace}

Primary task source: {task_origin}

Primary task:
{context.task_text.strip()}

Existing generated harness context:
{context.artifact_context}
{score_target}

Harness activation for {agent_name}:
- HarnessGym activates generated skills from .harnessgym/skills into .agents/skills for Codex and .claude/skills for Claude Code before each fresh attempt.
- HarnessGym activates generated MCP servers from .harnessgym/mcp manifests into .codex/config.toml for Codex and a generated Claude MCP config for Claude Code before each fresh attempt.
- For Codex exec sessions, if generated MCP tools are not exposed as native callables, use the workspace-local helper `python3 .harnessgym/runtime/mcp_call.py --server <server> --tool <tool> --arguments '<json-object>'`. This calls the generated MCP through HarnessGym telemetry and counts as concrete MCP tool use.
- If a relevant HarnessGym skill or MCP server is available, prefer using it early.

Attempt instructions:
- Work autonomously on the primary task until it is solved, blocked, or the process is interrupted by timeout.
- Use existing .harnessgym artifacts if they are relevant.
- At the start of the attempt, inspect the active HarnessGym skill/MCP context. If an MCP exposes benchmark, verifier, diagnostic, search, ranking, or rollback tools relevant to this task, use those tools early before doing manual investigation.
- If active MCP tools are relevant, make concrete MCP tool calls rather than only reading their files or mentioning them in prose. For Codex exec, prefer native MCP callables if present; otherwise use `.harnessgym/runtime/mcp_call.py`. Activation alone does not count as harness use.
- Do not write an ad hoc JSON-RPC MCP client or launch `.harnessgym/mcp/...` server files directly unless the native MCP callable and `mcp_call.py` helper both fail; those manual bypasses lose telemetry and do not count as verified harness tool use.
- If an MCP call fails, record the failure and then fall back to direct commands or local scripts.
- Prefer generated harness tools for repetitive measurement, hidden-state inspection, variant search, regression comparison, and final verification. Do not ignore an active harness and manually re-create the same workflow unless the harness is broken.
- Do not build new harness tooling during the attempt unless it is directly required to solve the primary task.
- If solved, run an objective verification command and write status "solved" with verified true.
- If this is an optimization task, run the benchmark/verifier, record the best numeric objective score in metrics.score, and keep improving until blocked, timed out, or a target is reached.
- For time-boxed optimization, once you have a verified candidate that improves the current workspace, apply/checkpoint that candidate and update result.json before spending more time searching for a better one. Reserve time for final verification and result.json writes.
- If the task exposes both fast/dev and final/held-out verifier modes, use the fast mode during iteration but run the final/held-out mode before claiming an improvement is robust.
- Keep or restore a last-known-good implementation before trying risky optimizations. Do not leave the workspace in a state that fails the objective verifier if a risky experiment has not passed.
- Record which generated artifacts or active MCP tools you used under `used_harness_artifacts` or `used_harness_tools` in the result JSON.
- Keep concise notes about blockers, missing tooling, slow investigation steps, and verification.
- Write or update the machine-readable JSON result at exactly:
  {context.result_path}

Use this JSON shape where practical:
{json.dumps(RESULT_SHAPE, indent=2)}
"""


def render_reflection_prompt(config: RunConfig, context: IterationContext, attempt_result: dict) -> str:
    agent_name = _agent_name(config)
    depth_guidance = _harness_depth_guidance(config.harness_depth)
    return f"""Continue in the same {agent_name} session for HarnessGym iteration {context.iteration}.

Reflect on the actual attempt you just performed. Based on that work, what skill, MCP server, verifier, fixture, script, docs, or tool would most improve solve time or make this task more solvable in the next fresh session?

Attempt result JSON currently contains:
{json.dumps(attempt_result, indent=2, sort_keys=True)}

Reflection instructions:
- Recommend one or two high-leverage additions.
- Choose the single best thing to build next.
- Prefer repo-local artifacts under .harnessgym/.
- Treat "single best thing" as a cohesive harness suite when that is highest leverage: it may include one skill, one MCP server, fixtures, tests, scripts, and docs that work together under .harnessgym/.
- If the task is a kernel, numerical, compiler, search, or verifier-driven task, strongly prefer an agent-native harness package: a skill under .harnessgym/skills plus a stdio MCP server under .harnessgym/mcp that exposes verifier/debugging/search commands.
- For follow-on iterations, prefer extending and hardening the existing harness suite over creating a disconnected note or wrapper. Add new diagnostic/search capability that the previous attempt genuinely lacked.
- For optimization tasks, explicitly identify benchmark modes, objective metrics, shape/seed assumptions, and overfitting risks. If a final or held-out evaluator exists, the selected improvement should make it easy for the next fresh session to run it.
- For MCP artifacts, include a JSON manifest named mcp.json, server.json, or harnessgym-mcp.json with at least: name, command, args, cwd, enabled_tools.
- MCP servers must speak stdio JSON-RPC with `Content-Length: ...\\r\\n\\r\\n` message framing. Newline-delimited JSON is not sufficient for Codex or Claude MCP activation.
- The selected improvement must include deterministic repo-local tests for the generated tooling itself. For MCP/tooling, plan a self-test that exercises Content-Length framing, initialize/tools/list/tools/call, at least one successful path, and at least one error or invalid-input path when practical.
- If the task or tooling is numerical, kernel, compiler, or benchmark related, the selected improvement must include numerical tests: known toy inputs, tolerance checks, fixed-seed randomized/property cases, and clear fast/dev versus final/held-out modes when such modes exist.
{depth_guidance}
- Update the result JSON at exactly:
  {context.result_path}
- Put the chosen item under reflection.selected_improvement using this shape:
  {json.dumps(RESULT_SHAPE["reflection"]["selected_improvement"], indent=2)}
"""


def render_build_prompt(
    config: RunConfig,
    context: IterationContext,
    selected_improvement: str,
) -> str:
    agent_name = _agent_name(config)
    depth_guidance = _harness_depth_guidance(config.harness_depth)
    return f"""Continue in the same {agent_name} session for HarnessGym iteration {context.iteration}.

Build the single highest-leverage HarnessGym improvement selected during reflection.

Selected improvement:
{selected_improvement}

Build instructions:
- Build only this one improvement.
- Make the artifact directly useful to a fresh agent session by including concise usage instructions and any task-specific invariants discovered from the attempt.
- Prefer an integrated harness suite when useful: a skill entrypoint plus an MCP server, tests, fixtures, history/cache files, and helper scripts are allowed as one cohesive improvement.
- If the task has dev and final/held-out verifier modes, include both in the generated skill/MCP/tooling and clearly distinguish "fast iteration" from "authoritative comparison".
- For optimization tasks, include concrete successful implementation patterns, failed traps, objective metric names, and any shape/seed assumptions. Avoid tooling that only optimizes a single dev case when a broader evaluator exists.
- If the selected improvement is a harness package, it may contain both a skill and an MCP server as one cohesive improvement.
- For skills, create a valid SKILL.md with frontmatter name and description under .harnessgym/skills/<skill-name>/.
- For MCP, create a stdio server under .harnessgym/mcp/<server-name>/ plus a manifest named mcp.json, server.json, or harnessgym-mcp.json. The manifest must include name, command, args, cwd, enabled_tools, and timeouts.
- The MCP server must implement Content-Length framed stdio JSON-RPC (`Content-Length: <bytes>\\r\\n\\r\\n<body>`) for initialize, notifications/initialized, tools/list, and tools/call. Validate it with a smoke test before updating the registry.
- MCP tools should return structured JSON text with stable fields. Avoid tools whose only output is prose.
- For deep harness work, cover this capability mix when practical: objective verifier runner, independent correctness/numerical checker, source/IR/assembly/trace analyzer, rollback-safe variant evaluator, sweep/autotune/search driver, history/regression comparator, and experiment ranker.
- Add comprehensive deterministic tests for all generated tooling. Put reusable tests under .harnessgym/tests/ when practical, and make them runnable from the workspace with no network dependency.
- For MCP servers, include a manifest `self_test` entry so HarnessGym can run the test during activation. Prefer `self_test: true` when `command + args + --self-test` works, or an explicit command list/dict when a separate test runner is clearer.
- MCP self-tests should cover Content-Length framing, initialize/tools/list/tools/call, at least one successful tool call, and at least one invalid-input/error path when practical. If tools mutate source files, test rollback/restore behavior.
- Self-tests must be portable to a copied fresh replay workspace. Do not hard-code the training workspace path, do not depend on benchmark timing noise, and do not assert exact winning variants or exact pass/fail sets for performance sweeps unless those outcomes are fixture-controlled. For sweep/search tools, assert stable schema, rollback, presence of expected candidate names, and structured handling of both successful and failed candidates.
- For numerical, kernel, compiler, or benchmark tooling, include numerical tests with known toy inputs, tolerance checks, fixed-seed randomized/property cases, and both fast/dev and final/held-out benchmark entrypoints when applicable.
- Record every generated-tooling test command and pass/fail status under `verification.tooling_tests` in the result JSON.
{depth_guidance}
- Store generated reusable artifacts under the appropriate .harnessgym/ directory:
  - .harnessgym/skills/
  - .harnessgym/mcp/
  - .harnessgym/tools/
  - .harnessgym/verifiers/
  - .harnessgym/fixtures/
  - .harnessgym/tests/
  - .harnessgym/docs/
  - .harnessgym/scripts/
- Update .harnessgym/registry.json with the artifact path, kind, description, and iteration.
- Update the result JSON at exactly:
  {context.result_path}
- Do not continue solving the original task except to validate that the new tooling works.
- Leave TODO notes for future MCP auto-registration or stronger evaluation if relevant.
"""


def render_repair_prompt(
    config: RunConfig,
    context: IterationContext,
    qualification_report: dict,
    repair_index: int,
) -> str:
    agent_name = _agent_name(config)
    depth_guidance = _harness_depth_guidance(config.harness_depth)
    return f"""Continue in the same {agent_name} session for HarnessGym iteration {context.iteration}.

HarnessGym fresh-workspace qualification failed after the build phase. Repair the generated harness artifact only.

Repair attempt: {repair_index}

Qualification failure report:
{json.dumps(qualification_report, indent=2, sort_keys=True)}

Repair instructions:
- Do not continue solving the original task.
- Fix the generated HarnessGym artifacts under .harnessgym/ so they activate in a fresh copied workspace.
- Prioritize the exact failure from the qualification report: MCP self-test failures, broken JSON-RPC framing, non-portable paths, missing fixtures, brittle benchmark assertions, or tools that return JSON-RPC errors.
- Keep tests comprehensive but portable. Self-tests must pass from a clean replay workspace with only task files plus .harnessgym artifacts copied in.
- If an MCP tool can fail for candidate-specific reasons, return structured tool results that include failed candidates instead of raising JSON-RPC errors for the whole sweep.
- Re-run the generated tooling tests and update result.json at exactly:
  {context.result_path}
- Record repair test commands under `verification.tooling_tests`.
{depth_guidance}
"""


def _harness_depth_guidance(harness_depth: str) -> str:
    if harness_depth != "deep":
        return "- Keep the generated artifact focused and small enough to build and validate within the phase timeout."
    return """- Deep harness mode is enabled: build capability-creating instrumentation, not just notes, aliases, or a benchmark wrapper.
- The chosen improvement should expose hidden task state or automate a search step that a vanilla coding agent would otherwise do manually.
- Prefer a multi-tool MCP server plus a short skill entrypoint. For kernel/compiler/hardware tasks, consider tools for assembly or IR dumps, vectorization/compiler diagnostics, layout or memory-traffic analysis, benchmark-history storage, final-mode repeated evaluation, source variant generation/sweeps, and regression comparison.
- If the task resembles TPU/GPU/custom-kernel work, think in the style of IR dump analyzers, DMA/descriptor inspectors, tensor-layout tools, trace summarizers, and objective-verifier dashboards.
- The artifact should have at least five genuinely useful executable tools when practical. Aim for a suite spanning observation, verification, search, rollback-safe mutation, history comparison, and next-experiment ranking.
- Each tool should remove work from the next fresh session: it should expose information, perform a safe search, verify a candidate, or summarize evidence faster than manual shell commands.
- Risky experiment tools should be rollback-safe: preserve the original source, benchmark candidate variants in temporary or restorable form, and only keep a variant after objective verification passes.
- Deep harness artifacts should ship with comprehensive self-tests. For numerical or kernel work, include tolerance-based toy cases, fixed-seed randomized cases, and tests that prevent benchmark-only overfitting.
- Maintain accumulated harness maturity across iterations: update existing skills/MCPs, add regression tests for previous tool behavior, preserve benchmark history, and document what the next attempt should try first.
- Avoid a documentation-only artifact unless it is paired with executable inspection or automation that materially changes the next fresh session's search space."""


def extract_selected_improvement(result_data: dict, fallback: str = "") -> str:
    reflection = result_data.get("reflection")
    if isinstance(reflection, dict):
        selected = reflection.get("selected_improvement")
        if selected:
            return json.dumps(selected, indent=2, sort_keys=True)
    selected = result_data.get("selected_improvement")
    if selected:
        return json.dumps(selected, indent=2, sort_keys=True)
    if fallback.strip():
        return fallback.strip()
    return "No structured selection found. Build the highest-leverage recommendation from the reflection transcript."
