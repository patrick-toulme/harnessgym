from __future__ import annotations

import json
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path

from harnessgym.artifacts import read_json, update_result
from harnessgym.config import RunConfig
from harnessgym.models import IterationContext, RunnerResult
from harnessgym.orchestrator import Orchestrator, _contains_artifact_reference, _contains_usage_token
from harnessgym.runners.base import Runner
from harnessgym.runners.fake_runner import FakeRunner


class ScoreMutatingRunner(Runner):
    def start_attempt(self, config: RunConfig, context: IterationContext, prompt: str) -> RunnerResult:
        started = time.monotonic()
        (context.workspace / "score.json").write_text(
            json.dumps({"status": "passed", "best_cycles": 4}),
            encoding="utf-8",
        )
        update_result(
            context.result_path,
            {
                "status": "incomplete",
                "verified": False,
                "used_harness_tools": ["score_probe"],
            },
        )
        return RunnerResult(
            phase="attempt",
            status="timeout",
            timed_out=True,
            duration_seconds=time.monotonic() - started,
            message="simulated timeout after mutating workspace",
        )

    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise AssertionError("attempt-only test should not reflect")

    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise AssertionError("attempt-only test should not build")


class SequenceScoreRunner(Runner):
    def __init__(self, scores: list[int]) -> None:
        self.scores = scores

    def start_attempt(self, config: RunConfig, context: IterationContext, prompt: str) -> RunnerResult:
        started = time.monotonic()
        score = self.scores[context.iteration - 1]
        (context.workspace / "kernel.txt").write_text(f"iteration-{context.iteration}\n", encoding="utf-8")
        (context.workspace / "score.json").write_text(
            json.dumps({"status": "passed", "best_cycles": score}),
            encoding="utf-8",
        )
        update_result(context.result_path, {"status": "incomplete", "verified": False})
        return RunnerResult(
            phase="attempt",
            status="timeout",
            timed_out=True,
            duration_seconds=time.monotonic() - started,
            message=f"simulated score {score}",
        )

    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise AssertionError("attempt-only test should not reflect")

    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise AssertionError("attempt-only test should not build")


class OutputUsageRunner(Runner):
    def start_attempt(self, config: RunConfig, context: IterationContext, prompt: str) -> RunnerResult:
        stdout_path = context.iteration_dir / "attempt.simulated.stdout.txt"
        stdout_path.write_text(
            "I used the generated MCP tool run_verifier from .harnessgym/mcp/kernel-tools.\n",
            encoding="utf-8",
        )
        update_result(context.result_path, {"status": "incomplete", "verified": False})
        return RunnerResult(
            phase="attempt",
            status="timeout",
            timed_out=True,
            stdout_path=str(stdout_path),
            duration_seconds=0.01,
        )

    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise AssertionError("attempt-only test should not reflect")

    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise AssertionError("attempt-only test should not build")


USAGE_MCP_SERVER = r'''
import json
import sys

if "--self-test" in sys.argv:
    raise SystemExit(0)

def read_msg():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        name, _, value = line.decode("ascii").partition(":")
        headers[name.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))

def write_msg(payload):
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()

while True:
    msg = read_msg()
    if msg is None:
        break
    method = msg.get("method")
    if method == "initialize":
        write_msg({"jsonrpc": "2.0", "id": msg.get("id"), "result": {"capabilities": {"tools": {}}}})
    elif method == "tools/list":
        write_msg({"jsonrpc": "2.0", "id": msg.get("id"), "result": {"tools": [{"name": "run_verifier"}]}})
'''


class PostAttemptTests(unittest.TestCase):
    def test_post_attempt_verifier_can_mark_blocked_attempt_solved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            verifier = workspace / "post_verify.py"
            verifier.write_text(
                "import json\nprint(json.dumps({'status': 'passed', 'score': 1}))\n",
                encoding="utf-8",
            )
            config = RunConfig.from_values(
                workspace=workspace,
                task_text="Solve with post verifier.",
                runner="fake",
                run_id="post-solved",
                post_attempt_command=f"{shlex.quote(sys.executable)} post_verify.py",
                post_attempt_timeout="5s",
            )

            results = Orchestrator(FakeRunner()).run(config)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "solved")
            self.assertTrue(results[0].solved)
            result_data = read_json(Path(results[0].result_path))
            self.assertEqual(result_data["verification"]["status"], "passed")
            self.assertEqual(result_data["metrics"]["score"], 1.0)

    def test_optimization_mode_records_improvement_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "score.json").write_text(
                json.dumps({"status": "passed", "best_cycles": 10}),
                encoding="utf-8",
            )
            score_script = workspace / "score.py"
            score_script.write_text(
                "from pathlib import Path\nprint(Path('score.json').read_text())\n",
                encoding="utf-8",
            )
            config = RunConfig.from_values(
                workspace=workspace,
                task_text="Reduce best_cycles.",
                iterations=1,
                attempt_timeout="1s",
                build_timeout="1s",
                runner="fake",
                run_id="post-optimization",
                stop_score=1,
                score_key="best_cycles",
                attempt_only=True,
                optimization_mode=True,
                post_attempt_command=f"{shlex.quote(sys.executable)} score.py",
                post_attempt_score_key="best_cycles",
                post_attempt_timeout="5s",
            )

            orchestrator = Orchestrator(ScoreMutatingRunner())
            results = orchestrator.run(config)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "improved")
            self.assertFalse(results[0].solved)
            result_data = read_json(Path(results[0].result_path))
            self.assertEqual(result_data["metrics"]["best_cycles"], 4.0)
            self.assertTrue(result_data["optimization"]["improved_from_baseline"])
            self.assertEqual(result_data["optimization"]["baseline_score"], 10.0)
            summary = read_json(workspace / ".harnessgym" / "runs" / "post-optimization" / "summary.json")
            self.assertTrue(summary["optimization"]["improved"])
            self.assertEqual(summary["optimization"]["best_score"], 4.0)
            self.assertEqual(summary["baseline_post_attempt"]["score"], 10.0)

    def test_optimization_mode_restores_best_checkpoint_after_later_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "kernel.txt").write_text("baseline\n", encoding="utf-8")
            (workspace / "score.json").write_text(
                json.dumps({"status": "passed", "best_cycles": 10}),
                encoding="utf-8",
            )
            score_script = workspace / "score.py"
            score_script.write_text(
                "from pathlib import Path\nprint(Path('score.json').read_text())\n",
                encoding="utf-8",
            )
            config = RunConfig.from_values(
                workspace=workspace,
                task_text="Reduce best_cycles.",
                iterations=2,
                attempt_timeout="1s",
                build_timeout="1s",
                runner="fake",
                run_id="restore-best",
                stop_score=1,
                score_key="best_cycles",
                attempt_only=True,
                optimization_mode=True,
                post_attempt_command=f"{shlex.quote(sys.executable)} score.py",
                post_attempt_score_key="best_cycles",
                post_attempt_timeout="5s",
            )

            results = Orchestrator(SequenceScoreRunner([4, 8])).run(config)

            self.assertEqual(len(results), 2)
            self.assertEqual((workspace / "kernel.txt").read_text(encoding="utf-8"), "iteration-1\n")
            summary = read_json(workspace / ".harnessgym" / "runs" / "restore-best" / "summary.json")
            self.assertEqual(summary["optimization"]["best_score"], 4.0)
            self.assertTrue(summary["best_checkpoint"]["restored"])
            self.assertEqual(summary["best_checkpoint"]["restore"]["iteration"], 1)
            self.assertEqual(summary["best_checkpoint"]["restore"]["post_restore"]["score"], 4.0)
            final_data = read_json(Path(results[-1].result_path))
            self.assertEqual(final_data["optimization"]["restored_best_checkpoint"]["score"], 4.0)

    def test_optimization_mode_can_leave_final_state_when_restore_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "kernel.txt").write_text("baseline\n", encoding="utf-8")
            (workspace / "score.json").write_text(
                json.dumps({"status": "passed", "best_cycles": 10}),
                encoding="utf-8",
            )
            score_script = workspace / "score.py"
            score_script.write_text(
                "from pathlib import Path\nprint(Path('score.json').read_text())\n",
                encoding="utf-8",
            )
            config = RunConfig.from_values(
                workspace=workspace,
                task_text="Reduce best_cycles.",
                iterations=2,
                attempt_timeout="1s",
                build_timeout="1s",
                runner="fake",
                run_id="no-restore-best",
                stop_score=1,
                score_key="best_cycles",
                attempt_only=True,
                optimization_mode=True,
                restore_best=False,
                post_attempt_command=f"{shlex.quote(sys.executable)} score.py",
                post_attempt_score_key="best_cycles",
                post_attempt_timeout="5s",
            )

            Orchestrator(SequenceScoreRunner([4, 8])).run(config)

            self.assertEqual((workspace / "kernel.txt").read_text(encoding="utf-8"), "iteration-2\n")
            summary = read_json(workspace / ".harnessgym" / "runs" / "no-restore-best" / "summary.json")
            self.assertFalse(summary["best_checkpoint"]["enabled"])
            self.assertEqual(summary["best_checkpoint"]["reason"], "disabled_by_config")

    def test_harness_usage_can_be_inferred_from_attempt_output_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            mcp_dir = workspace / ".harnessgym" / "mcp" / "kernel-tools"
            mcp_dir.mkdir(parents=True)
            (mcp_dir / "server.py").write_text(USAGE_MCP_SERVER, encoding="utf-8")
            (mcp_dir / "mcp.json").write_text(
                json.dumps(
                    {
                        "name": "kernel-tools",
                        "command": sys.executable,
                        "args": [".harnessgym/mcp/kernel-tools/server.py"],
                        "cwd": ".",
                        "enabled_tools": ["run_verifier"],
                        "self_test": True,
                    }
                ),
                encoding="utf-8",
            )
            config = RunConfig.from_values(
                workspace=workspace,
                task_text="Use generated harness if relevant.",
                iterations=1,
                runner="fake",
                run_id="usage-inference",
                attempt_only=True,
            )

            results = Orchestrator(OutputUsageRunner()).run(config)

            result_data = read_json(Path(results[0].result_path))
            usage = result_data["harness_usage"]
            self.assertIn("run_verifier", usage["inferred_harness_tools"])
            self.assertIn("run_verifier", usage["used_active_tools"])
            self.assertGreaterEqual(usage["inferred_tool_use_count"], 1)


class ContainsUsageTokenTests(unittest.TestCase):
    def test_prose_does_not_match_short_tool_name(self) -> None:
        self.assertFalse(_contains_usage_token("i will run the tests", "run"))
        self.assertFalse(_contains_usage_token("i will run the tests", "test"))

    def test_mcp_pattern_matches(self) -> None:
        self.assertTrue(_contains_usage_token("calling mcp__kernel-tools__run_verifier", "run_verifier"))

    def test_tools_call_with_quoted_name_matches(self) -> None:
        haystack = 'tools/call "run_verifier" with args'
        self.assertTrue(_contains_usage_token(haystack, "run_verifier"))

    def test_tool_name_near_tool_keyword_matches(self) -> None:
        self.assertTrue(_contains_usage_token("I used the generated MCP tool run_verifier from .harnessgym/mcp/kernel-tools.", "run_verifier"))

    def test_artifact_path_basename_matches_near_tool(self) -> None:
        self.assertTrue(_contains_usage_token("used tool from .harnessgym/mcp/kernel-tools", ".harnessgym/mcp/kernel-tools"))


class ContainsArtifactReferenceTests(unittest.TestCase):
    def test_exact_artifact_path_matches_without_tool_context(self) -> None:
        self.assertTrue(
            _contains_artifact_reference(
                "I used .harnessgym/skills/foo/SKILL.md during the attempt.",
                ".harnessgym/skills/foo/SKILL.md",
            )
        )

    def test_artifact_basename_matches_without_tool_context(self) -> None:
        self.assertTrue(_contains_artifact_reference("I read SKILL.md during the attempt.", ".harnessgym/skills/foo/SKILL.md"))


if __name__ == "__main__":
    unittest.main()
