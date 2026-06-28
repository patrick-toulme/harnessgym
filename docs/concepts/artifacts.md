# Artifacts & Registry

Everything HarnessGym generates is a **repo-local artifact** under
`.harnessgym/`. The registry tracks what exists, what's been qualified, and
what's been quarantined. This page is the reference for that layout.

## Artifact kinds

The build phase can produce any of eight kinds, each with its own directory
(`artifacts.ARTIFACT_DIRS`):

| Kind | Directory | Typical contents |
| --- | --- | --- |
| `skill` | `.harnessgym/skills/<name>/` | `SKILL.md` with frontmatter `name` + `description` |
| `mcp` | `.harnessgym/mcp/<name>/` | stdio server + a manifest (`mcp.json` / `server.json` / `harnessgym-mcp.json`) |
| `tool` | `.harnessgym/tools/` | standalone executable helpers |
| `verifier` | `.harnessgym/verifiers/` | correctness / objective checkers |
| `fixture` | `.harnessgym/fixtures/` | inputs, golden outputs, seeds |
| `test` | `.harnessgym/tests/` | deterministic tests for the generated tooling |
| `docs` | `.harnessgym/docs/` | notes, invariants, gotchas |
| `script` | `.harnessgym/scripts/` | reproducible run scripts |

The richest artifact is usually a **harness suite**: a short skill entrypoint
plus a multi-tool MCP server plus its tests and fixtures, all built in one
iteration as a single cohesive improvement.

## On-disk layout

```text
.harnessgym/
├── registry.json          # the inventory (source of truth for what's active)
├── activation.json        # latest activation result + quality gate
├── mcp_calls.jsonl        # every generated MCP tools/call (telemetry)
├── runtime/mcp_call.py    # helper for calling generated MCPs from a session
├── skills/  mcp/  tools/  verifiers/  fixtures/  tests/  docs/  scripts/
├── task_state/initial/    # snapshot for --task-state reset
└── runs/<run_id>/
    ├── run_config.json
    ├── summary.json
    └── iterations/<n>/
        ├── result.json
        ├── activation.json
        ├── post_build_activation.json
        ├── <phase>.prompt.txt / .stdout.txt / .stderr.txt / .transcript.txt
        ├── qualification/attempt_<k>/qualification.json
        └── repair_<k>/...
```

`runs/` holds the per-run, per-iteration logs and is append-only history. The
artifact directories (`skills/`, `mcp/`, …) hold the reusable bundle that
travels to future runs.

## The registry

`.harnessgym/registry.json` is a small JSON document
(`models.Registry`/`models.Artifact`):

```json
{
  "version": 1,
  "updated_at": "<iso-8601 timestamp>",
  "artifacts": [
    {
      "id": "mcp:.harnessgym/mcp/cpu_attention_autotune/harnessgym-mcp.json",
      "kind": "mcp",
      "path": ".harnessgym/mcp/cpu_attention_autotune/harnessgym-mcp.json",
      "description": "...",
      "iteration": 3,
      "created_at": "<iso-8601 timestamp>",
      "metadata": {
        "qualification": { "status": "passed", "report_path": "...", "iteration": 3 }
      }
    }
  ],
  "metadata": {}
}
```

Each artifact carries an `id` (`<kind>:<path>`), the workspace-relative `path`,
a `description`, the `iteration` that created it, and a `metadata` bag that
holds qualification/quarantine status.

### Synced from the filesystem

You don't register artifacts by hand. After each build, HarnessGym walks the
artifact directories and reconciles the registry with what's actually on disk
(`sync_registry_from_files`):

- `skill` entries are only registered for `SKILL.md` files.
- `mcp` entries are only registered for a recognized manifest filename.
- `__pycache__`, `.pyc`, and `.pyo` are ignored.

So the registry never claims a tool that isn't there, and a tool dropped into
the right directory is picked up automatically.

### Active vs quarantined

The registry distinguishes **active** artifacts (advertised to attempts and
activated) from **quarantined** ones (`registry.artifact_is_quarantined`). A
quarantined artifact keeps its files but is hidden from every future attempt
prompt and never injected into a runner session — see
[Qualification](qualification.md). The attempt prompt renders only the active
set, plus a count of how many are quarantined and where to find their repair
evidence.

## Using the framework objects

The registry and artifact models are part of the public API:

```python
from harnessgym import Registry, Artifact
from harnessgym.registry import load_registry, active_artifacts

registry = load_registry(workspace)          # reads .harnessgym/registry.json
for artifact in active_artifacts(registry):  # skips quarantined
    print(artifact.kind, artifact.path, artifact.description)
```

[**Qualification →**](qualification.md){ .md-button }
[**Activation →**](activation.md){ .md-button }
