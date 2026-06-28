# Install

`HarnessGym` keeps its core dependency-free. Pick the install that matches what
you want to do.

| Command | What you get |
| --- | --- |
| `pip install harnessgym` | orchestrator, registry, qualification, activation, replay — stdlib only |
| `pip install -e ".[dev]"` | + `build`, `pytest`, `twine` and the bundled examples (source checkout) |
| `pip install -e ".[docs]"` | + `mkdocs-material` for building this site |

## Base package

```bash
python -m pip install harnessgym
harnessgym --help
```

The core package has **zero third-party runtime dependencies**. Everything the
orchestrator needs — the five-phase loop, the registry, fresh-workspace
qualification, runner-native activation, MCP telemetry, and replay/compare — is
pure Python stdlib. Requires **Python 3.10+**.

## Source checkout (with tests and examples)

```bash
git clone https://github.com/patrick-toulme/harnessgym.git
cd harnessgym
python -m pip install -e ".[dev]"
python -m pytest
```

This is the install you want if you plan to run the bundled example tasks under
`examples/`, since most of them ship their own benchmark and verifier scripts.

## Runner prerequisites

HarnessGym shells out to whichever agent CLI you select with `--runner`:

=== "exec (Codex)"

    Requires the `codex` CLI on `PATH` (configurable with `--codex-bin`).
    This is the MVP backend.

=== "claude (Claude Code)"

    Requires the `claude` CLI on `PATH` (configurable with `--claude-bin`).
    Optional: `--claude-model sonnet|opus`, `--claude-max-budget-usd`.

=== "fake (offline)"

    No agent account, no network. Deterministic — used for tests and the
    install smoke check.

Some bundled examples additionally need local tooling: NumPy, a C/C++ compiler,
PyTorch, Triton, or access to a remote GPU. Each example's `task.md` states its
requirements.

## Docs tooling

The website *is* the docs. To build or serve it locally:

```bash
pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site into ./site
```

The `[docs]` extra pulls in `mkdocs-material[imaging]` (the `imaging` extra
powers social-card generation) and `pymdown-extensions`. On Linux the social
plugin also needs the Cairo/FreeType system libraries; the
[deploy workflow](https://github.com/patrick-toulme/harnessgym/blob/main/.github/workflows/deploy-docs.yml)
installs them.

## Verify the install

The fastest end-to-end check needs no agent account — run the offline fake
runner on the bundled numerical-debug demo:

```bash
harnessgym run \
  --task examples/numerical_debug_task/task.md \
  --workspace examples/numerical_debug_task \
  --iterations 2 \
  --attempt-timeout 10s \
  --build-timeout 10s \
  --runner fake
```

You should see HarnessGym block the first attempt, build a probe tool under
`.harnessgym/tools/`, update the registry, start a fresh second attempt with
that context, apply the demo fix, and record a verified result.

[**Getting Started →**](getting-started.md){ .md-button }
