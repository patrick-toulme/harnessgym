# Qualification

Qualification is the gate that separates *"the agent wrote some files"* from
*"a fresh session can actually use this."* It is on by default; disable it only
to debug the framework with `--no-qualify-artifacts`.

The rule: **a generated artifact is not promoted until it activates and
self-tests in a clean copy of the task workspace** — one that contains only the
original task files plus the `.harnessgym/` bundle, nothing else from the dirty
training tree.

## Why a clean room

During an attempt the workspace accumulates scratch files, environment state,
absolute paths, and half-finished experiments. A tool that "works" there might
secretly depend on all of that. The only honest test of reusability is to drop
the tool into a workspace that looks like what the *next* fresh session will
see. That is exactly what qualification builds.

## The mechanism

At the very start of the run, HarnessGym snapshots the clean task workspace
(`capture_clean_workspace_template`), excluding `.git`, `.harnessgym`,
`.codex`, `.agents`, build dirs, and compiled objects.

Then, after each build phase (`qualify_generated_harness`):

1. **Copy the clean template** into
   `runs/<run_id>/iterations/<n>/qualification/attempt_<k>/workspace`.
2. **Copy in only the reusable `.harnessgym/` bundle** (`copy_reusable_harness`)
   — the artifact directories plus `registry.json`, ignoring `__pycache__`.
3. **Activate** the generated skills and MCP servers in that fresh workspace
   (the same [activation](activation.md) path used for a real attempt).
4. **Run MCP self-tests** and evaluate the quality gate.

The result is written to `qualification.json` with `status: passed | failed`,
the copied artifacts, the full activation report, the quality gate, and any
failed artifact paths.

## The quality gate

An MCP server passes its gate (`activation._mcp_quality_gate`) only when all of:

- **smoke** — `initialize` + `tools/list` succeed, and
- **self-test** — the manifest's `self_test` runs and passes, and
- **tool inventory** — the server exposes a non-empty tool set.

The overall gate is `passed` only if every advertised server passes and at
least one tool is active. The gate records `active_mcp_count`,
`inactive_mcp_count`, and `active_tool_count`, which flow into the attempt
prompt and `summary.json`.

## Repair, then quarantine

If qualification fails, HarnessGym doesn't give up immediately — and it doesn't
silently keep the broken tool either.

```text
build → qualify ──pass──▶ mark qualified, continue
           │
          fail
           │
           ▼
   repair (same session, up to --artifact-repair-attempts)
           │
        still fail
           │
           ▼
       quarantine (hide from future attempts, keep for evidence)
```

- **Repair.** The exact qualification failure report is sent back into the same
  session (`render_repair_prompt`) with a narrow instruction: *fix the
  generated artifact so it activates in a fresh copied workspace — do not keep
  solving the original task.* This repeats up to `--artifact-repair-attempts`
  times (default `1`). The repair prompt specifically calls out the common
  failure modes: MCP self-test failures, broken Content-Length framing,
  non-portable absolute paths, missing fixtures, brittle benchmark assertions,
  and tools that raise JSON-RPC errors for a whole sweep instead of returning
  structured failed-candidate results.
- **Quarantine.** Anything still failing is quarantined in `registry.json`
  (`quarantine_artifacts`): the files remain on disk with quarantine metadata
  and the failure-report path, but the artifact is hidden from attempt prompts
  and never activated. The next iteration's repair build has the concrete
  evidence to fix it; a broken harness can never pollute a fresh attempt.

## Writing qualifiable artifacts

The build/reflection prompts steer the agent toward artifacts that survive this
gate, and your task file can reinforce it. The traits that matter:

- **Content-Length framed stdio JSON-RPC.** `Content-Length: <bytes>\r\n\r\n<body>`
  for `initialize`, `notifications/initialized`, `tools/list`, and `tools/call`.
  Newline-delimited JSON fails activation.
- **A manifest `self_test`.** `true` (runs `command + args + --self-test`), or a
  command string/list/object pointing at a separate runner.
- **Portable self-tests.** No hard-coded training-workspace path, no dependence
  on benchmark timing noise, no asserting exact winning variants for a
  stochastic sweep. Assert stable schema, rollback behavior, expected candidate
  names, and structured handling of both successful and failed candidates.
- **Numerical coverage** where relevant: toy inputs, tolerance checks,
  fixed-seed randomized cases, and fast/dev vs final/held-out modes.

[**Activation →**](activation.md){ .md-button }
[**MCP Telemetry →**](telemetry.md){ .md-button }
