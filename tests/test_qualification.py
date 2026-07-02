from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from harnessgym.artifacts import update_result
from harnessgym.config import RunConfig
from harnessgym.models import Artifact, IterationContext, Registry, RunnerResult
from harnessgym.orchestrator import Orchestrator
from harnessgym.qualification import _failed_artifact_paths
from harnessgym.registry import load_registry
from harnessgym.runners.base import Runner


MCP_TEMPLATE = r'''
import json
import sys

PASS_SELF_TEST = {pass_self_test}

if "--self-test" in sys.argv:
    print(json.dumps({{"status": "passed" if PASS_SELF_TEST else "failed"}}))
    raise SystemExit(0 if PASS_SELF_TEST else 1)


def read_msg():
    headers = {{}}
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
    sys.stdout.buffer.write(f"Content-Length: {{len(body)}}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


while True:
    msg = read_msg()
    if msg is None:
        break
    method = msg.get("method")
    if method == "initialize":
        write_msg({{"jsonrpc": "2.0", "id": msg.get("id"), "result": {{"capabilities": {{"tools": {{}}}}}}}})
    elif method == "tools/list":
        write_msg({{"jsonrpc": "2.0", "id": msg.get("id"), "result": {{"tools": [{{"name": "inspect_ir"}}]}}}})
'''


class RepairingMcpRunner(Runner):
    def __init__(self, *, repair_succeeds: bool = True) -> None:
        self.build_calls = 0
        self.repair_succeeds = repair_succeeds
        self.attempt_prompts: list[str] = []

    def start_attempt(self, config: RunConfig, context: IterationContext, prompt: str) -> RunnerResult:
        self.attempt_prompts.append(prompt)
        update_result(
            context.result_path,
            {
                "status": "blocked",
                "verified": False,
                "missing_tooling": ["Need generated MCP diagnostics."],
            },
        )
        return self._result(context, "attempt", prompt)

    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        update_result(
            context.result_path,
            {
                "reflection": {
                    "selected_improvement": {
                        "kind": "mcp",
                        "name": "ir-tools",
                        "target_path": ".harnessgym/mcp/ir-tools/harnessgym-mcp.json",
                    }
                }
            },
        )
        return self._result(context, "reflection", prompt, session_id=session_id)

    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        self.build_calls += 1
        mcp_dir = context.workspace / ".harnessgym" / "mcp" / "ir-tools"
        mcp_dir.mkdir(parents=True, exist_ok=True)
        pass_self_test = self.repair_succeeds and self.build_calls > 1
        (mcp_dir / "server.py").write_text(
            MCP_TEMPLATE.format(pass_self_test="True" if pass_self_test else "False"),
            encoding="utf-8",
        )
        (mcp_dir / "harnessgym-mcp.json").write_text(
            json.dumps(
                {
                    "name": "ir-tools",
                    "command": "python3",
                    "args": [".harnessgym/mcp/ir-tools/server.py"],
                    "cwd": ".",
                    "enabled_tools": ["inspect_ir"],
                    "self_test": {
                        "command": "python3",
                        "args": [".harnessgym/mcp/ir-tools/server.py", "--self-test"],
                        "timeout_seconds": 5,
                    },
                }
            ),
            encoding="utf-8",
        )
        update_result(context.result_path, {"status": "tooling_built"})
        return self._result(context, "build", prompt, session_id=session_id)

    def _result(
        self,
        context: IterationContext,
        phase: str,
        prompt: str,
        session_id: str | None = None,
    ) -> RunnerResult:
        started = time.monotonic()
        context.iteration_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = context.iteration_dir / f"{phase}.prompt.txt"
        stdout_path = context.iteration_dir / f"{phase}.stdout.txt"
        stderr_path = context.iteration_dir / f"{phase}.stderr.txt"
        transcript_path = context.iteration_dir / f"{phase}.transcript.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        effective_session = session_id or "repair-session"
        transcript_path.write_text(f"session_id: {effective_session}\n", encoding="utf-8")
        return RunnerResult(
            phase=phase,
            status="completed",
            session_id=effective_session,
            return_code=0,
            duration_seconds=time.monotonic() - started,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            transcript_path=str(transcript_path),
            prompt_path=str(prompt_path),
        )


class QualificationTests(unittest.TestCase):
    def test_failed_fresh_qualification_triggers_same_session_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "task.md").write_text("Build IR tooling.\n", encoding="utf-8")
            config = RunConfig.from_values(
                workspace=workspace,
                task="task.md",
                iterations=1,
                runner="fake",
                build_timeout="5s",
                artifact_repair_attempts=1,
            )
            runner = RepairingMcpRunner()
            orchestrator = Orchestrator(runner=runner)

            orchestrator.run(config)

            self.assertEqual(runner.build_calls, 2)
            registry = load_registry(workspace)
            artifact = registry.get_artifact("mcp:.harnessgym/mcp/ir-tools/harnessgym-mcp.json")
            self.assertIsNotNone(artifact)
            assert artifact is not None
            self.assertEqual(artifact.metadata["qualification"]["status"], "passed")
            self.assertNotIn("quarantine", artifact.metadata)
            activation = json.loads((workspace / ".harnessgym" / "activation.json").read_text(encoding="utf-8"))
            self.assertEqual(activation["quality_gate"]["status"], "passed")
            self.assertEqual(activation["quality_gate"]["active_tool_count"], 1)
            result_path = (
                workspace
                / ".harnessgym"
                / "runs"
                / orchestrator.run_id
                / "iterations"
                / "1"
                / "result.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["artifact_qualification"]["status"], "passed")

    def test_failed_qualification_quarantines_artifact_from_next_attempt_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "task.md").write_text("Build IR tooling.\n", encoding="utf-8")
            config = RunConfig.from_values(
                workspace=workspace,
                task="task.md",
                iterations=2,
                runner="fake",
                build_timeout="5s",
                artifact_repair_attempts=0,
            )
            runner = RepairingMcpRunner(repair_succeeds=False)
            orchestrator = Orchestrator(runner=runner)

            orchestrator.run(config)

            registry = load_registry(workspace)
            artifact = registry.get_artifact("mcp:.harnessgym/mcp/ir-tools/harnessgym-mcp.json")
            self.assertIsNotNone(artifact)
            assert artifact is not None
            self.assertEqual(artifact.metadata["quarantine"]["status"], "quarantined")
            self.assertGreaterEqual(len(runner.attempt_prompts), 2)
            self.assertIn("quarantined artifact(s) are hidden", runner.attempt_prompts[1])
            self.assertNotIn("- mcp: .harnessgym/mcp/ir-tools/harnessgym-mcp.json", runner.attempt_prompts[1])

    def test_qualification_failed_paths_does_not_quarantine_prior_iteration_artifacts(self) -> None:
        orchestrator = Orchestrator()
        registry = Registry()
        a1 = Artifact(
            id="mcp:server1.py",
            kind="mcp",
            path=".harnessgym/mcp/server1.py",
            description="iter1",
            iteration=1,
            metadata={"qualification": {"status": "passed", "iteration": 1}},
        )
        a2 = Artifact(
            id="mcp:server2.py",
            kind="mcp",
            path=".harnessgym/mcp/server2.py",
            description="iter2",
            iteration=2,
            metadata={},
        )
        registry.artifacts.extend([a1, a2])

        # Case 1: iter-2 artifact exists, report failed with empty failed_artifacts.
        # Only the iter-2 artifact should be returned; iter-1 preserved.
        report = {"status": "failed", "failed_artifacts": [], "quality_gate": {"status": "failed"}}
        failed = orchestrator._qualification_failed_paths(report, registry, iteration=2)
        self.assertIn(".harnessgym/mcp/server2.py", failed)
        self.assertNotIn(".harnessgym/mcp/server1.py", failed)

        # Case 2: no iter-2 artifacts (build produced nothing), report failed.
        # The fallback must NOT return all artifacts — return empty.
        registry.artifacts = [a for a in registry.artifacts if a.iteration != 2]
        failed = orchestrator._qualification_failed_paths(report, registry, iteration=2)
        self.assertEqual(failed, [])
        self.assertNotIn(".harnessgym/mcp/server1.py", failed)

    def test_failed_artifact_paths_does_not_return_all_on_gate_failure(self) -> None:
        # When the quality gate fails but no specific inactive server is identified,
        # _failed_artifact_paths must return only the inactive servers, not all artifacts.
        activation = {
            "mcp_servers": [],
            "quality_gate": {"status": "failed"},
        }
        registry = Registry()
        registry.artifacts.extend([
            Artifact(id="mcp:a", kind="mcp", path=".harnessgym/mcp/a.json", description="a", iteration=1, metadata={}),
            Artifact(id="mcp:b", kind="mcp", path=".harnessgym/mcp/b.json", description="b", iteration=1, metadata={}),
        ])
        failed = _failed_artifact_paths(activation, registry)
        self.assertEqual(failed, [])
