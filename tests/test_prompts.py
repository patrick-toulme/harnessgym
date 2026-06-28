import tempfile
import unittest
import json
from pathlib import Path

from harnessgym.config import RunConfig
from harnessgym.models import Artifact, IterationContext, Registry
from harnessgym.prompts import (
    extract_selected_improvement,
    render_artifact_context,
    render_attempt_prompt,
    render_build_prompt,
    render_reflection_prompt,
)


class PromptTests(unittest.TestCase):
    def test_prompt_rendering_includes_required_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            registry = Registry(
                artifacts=[
                    Artifact(
                        id="tool:.harnessgym/tools/probe.py",
                        kind="tool",
                        path=".harnessgym/tools/probe.py",
                        description="Probe tool",
                        iteration=1,
                    )
                ]
            )
            docs_path = tmp_path / ".harnessgym" / "docs" / "brief.md"
            docs_path.parent.mkdir(parents=True)
            docs_path.write_text("Exact recurrence notes.\n", encoding="utf-8")
            registry.artifacts.append(
                Artifact(
                    id="docs:.harnessgym/docs/brief.md",
                    kind="docs",
                    path=".harnessgym/docs/brief.md",
                    description="Brief",
                    iteration=1,
                )
            )
            test_path = tmp_path / ".harnessgym" / "tests" / "test_probe.py"
            test_path.parent.mkdir(parents=True)
            test_path.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
            registry.artifacts.append(
                Artifact(
                    id="test:.harnessgym/tests/test_probe.py",
                    kind="test",
                    path=".harnessgym/tests/test_probe.py",
                    description="Probe tests",
                    iteration=1,
                )
            )
            activation_path = tmp_path / ".harnessgym" / "activation.json"
            activation_path.write_text(
                json.dumps(
                    {
                        "warnings": [],
                        "mcp_servers": [
                            {
                                "name": "kernel-suite",
                                "enabled_tools": ["run_benchmark", "rank_experiments"],
                                "smoke": {"status": "passed", "tools": ["run_benchmark", "rank_experiments"]},
                                "self_test": {"status": "passed"},
                            }
                        ],
                        "skills": [
                            {
                                "activated_path": str(tmp_path / ".agents" / "skills" / "kernel-suite"),
                                "artifact_path": ".harnessgym/skills/kernel-suite",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = RunConfig.from_values(workspace=tmp_path, task_text="Fix the task.", runner="fake")
            iteration_dir = tmp_path / ".harnessgym" / "runs" / "run" / "iterations" / "1"
            context = IterationContext(
                run_id="run",
                iteration=1,
                workspace=tmp_path,
                harness_dir=tmp_path / ".harnessgym",
                run_dir=tmp_path / ".harnessgym" / "runs" / "run",
                iteration_dir=iteration_dir,
                result_path=iteration_dir / "result.json",
                registry=registry,
                task_text=config.task_text,
                artifact_context=render_artifact_context(registry, tmp_path),
            )

            attempt = render_attempt_prompt(config, context)
            self.assertIn("Work autonomously on the primary task", attempt)
            self.assertIn("Do not build new harness tooling", attempt)
            self.assertIn(".harnessgym/tools/probe.py", attempt)
            self.assertIn("Exact recurrence notes.", attempt)
            self.assertIn("test_probe", attempt)
            self.assertIn("MCP `kernel-suite` active", attempt)
            self.assertIn("rank_experiments", attempt)
            self.assertIn(str(context.result_path), attempt)
            self.assertIn("use those tools early", attempt)
            self.assertIn("concrete MCP tool calls", attempt)
            self.assertIn("Activation alone does not count", attempt)
            self.assertIn("used_harness_tools", attempt)

            score_config = RunConfig.from_values(
                workspace=tmp_path,
                task_text="Optimize.",
                runner="fake",
                stop_score=2.0,
                score_key="best_ms",
            )
            score_attempt = render_attempt_prompt(score_config, context)
            self.assertIn("Optimization stop target", score_attempt)
            self.assertIn("best_ms", score_attempt)

            reflection = render_reflection_prompt(config, context, {"status": "blocked"})
            self.assertIn("same Codex session", reflection)
            self.assertIn("selected_improvement", reflection)
            self.assertIn("agent-native harness package", reflection)
            self.assertIn("Content-Length", reflection)
            self.assertIn("deterministic repo-local tests", reflection)
            self.assertIn("fixed-seed randomized", reflection)
            self.assertIn("cohesive harness suite", reflection)
            self.assertIn("extending and hardening the existing harness suite", reflection)

            selected = extract_selected_improvement(
                {
                    "reflection": {
                        "selected_improvement": {
                            "kind": "tool",
                            "name": "probe",
                            "target_path": ".harnessgym/tools/probe.py",
                        }
                    }
                }
            )
            build = render_build_prompt(config, context, selected)
            self.assertIn("Build only this one improvement", build)
            self.assertIn("directly useful to a fresh agent session", build)
            self.assertIn(".harnessgym/registry.json", build)
            self.assertIn("Content-Length framed stdio JSON-RPC", build)
            self.assertIn(".harnessgym/tests/", build)
            self.assertIn("self_test", build)
            self.assertIn("verification.tooling_tests", build)
            self.assertIn("rollback-safe variant evaluator", build)
            self.assertIn("history/regression comparator", build)
            self.assertIn("portable to a copied fresh replay workspace", build)
            self.assertIn("do not assert exact winning variants", build)

            deep_config = RunConfig.from_values(
                workspace=tmp_path,
                task_text="Optimize a kernel.",
                runner="fake",
                harness_depth="deep",
            )
            deep_reflection = render_reflection_prompt(deep_config, context, {"status": "blocked"})
            deep_build = render_build_prompt(deep_config, context, selected)
            self.assertIn("Deep harness mode is enabled", deep_reflection)
            self.assertIn("assembly or IR dumps", deep_build)
            self.assertIn("source variant generation/sweeps", deep_build)
            self.assertIn("rollback-safe", deep_build)
            self.assertIn("comprehensive self-tests", deep_build)
            self.assertIn("at least five", deep_build)
            self.assertIn("harness maturity", deep_build)

            claude_config = RunConfig.from_values(workspace=tmp_path, task_text="Fix the task.", runner="claude")
            claude_attempt = render_attempt_prompt(claude_config, context)
            claude_reflection = render_reflection_prompt(claude_config, context, {"status": "blocked"})
            self.assertIn("Harness activation for Claude Code", claude_attempt)
            self.assertIn(".claude/skills", claude_attempt)
            self.assertIn("generated Claude MCP config", claude_attempt)
            self.assertIn("same Claude Code session", claude_reflection)
