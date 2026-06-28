#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import select
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / ".harnessgym" / "mcp" / "tensor-plan-server" / "tensor_plan_server.py"
MANIFEST = ROOT / ".harnessgym" / "mcp" / "tensor-plan-server" / "harnessgym-mcp.json"
FIXTURE_NAME = "tensor_plan_attempt1_best"
ITER2_FIXTURE_NAME = "tensor_plan_iteration2_neighborhood_best"

sys.path.insert(0, str(ROOT))
import benchmark  # type: ignore  # noqa: E402


def encode_message(message: dict[str, Any]) -> bytes:
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def read_message(stream: BinaryIO, timeout: float = 5.0) -> dict[str, Any]:
    ready, _, _ = select.select([stream], [], [], timeout)
    if not ready:
        raise TimeoutError("timed out waiting for framed MCP response")
    headers: dict[str, str] = {}
    line = stream.readline()
    while line not in {b"\r\n", b"\n", b""}:
        text = line.decode("ascii").strip()
        if ":" in text:
            key, value = text.split(":", 1)
            headers[key.lower()] = value.strip()
        line = stream.readline()
    length = int(headers["content-length"])
    return json.loads(stream.read(length).decode("utf-8"))


def call_rpc(proc: subprocess.Popen[bytes], request_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(encode_message({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}))
    proc.stdin.flush()
    return read_message(proc.stdout)


def notify(proc: subprocess.Popen[bytes], method: str, params: dict[str, Any] | None = None) -> None:
    assert proc.stdin is not None
    proc.stdin.write(encode_message({"jsonrpc": "2.0", "method": method, "params": params or {}}))
    proc.stdin.flush()


def tool_payload(response: dict[str, Any]) -> dict[str, Any]:
    text = response["result"]["content"][0]["text"]
    return json.loads(text)


class ServerProcess:
    def __init__(self, cwd: Path, server: Path) -> None:
        self.cwd = cwd
        self.server = server
        self.proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> subprocess.Popen[bytes]:
        self.proc = subprocess.Popen(
            [sys.executable, str(self.server)],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return self.proc

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=3)


class TensorPlanMcpTests(unittest.TestCase):
    def test_manifest_is_portable_and_declares_self_test(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["name"], "tensor-plan-server")
        self.assertEqual(manifest["command"], "python3")
        self.assertEqual(manifest["cwd"], ".")
        self.assertTrue(manifest["self_test"])
        self.assertIn("search_plans", manifest["enabled_tools"])
        self.assertIn("local_neighborhood_search", manifest["enabled_tools"])
        self.assertIn("bounded_exhaustive_search", manifest["enabled_tools"])
        self.assertIn("apply_best_verified", manifest["enabled_tools"])
        self.assertIn("timeouts", manifest)
        self.assertFalse(os.path.isabs(manifest["args"][0]))

    def test_content_length_initialize_tools_and_error_path(self) -> None:
        with ServerProcess(ROOT, SERVER) as proc:
            init = call_rpc(proc, 1, "initialize", {"clientInfo": {"name": "self-test", "version": "1"}})
            self.assertEqual(init["result"]["serverInfo"]["name"], "tensor-plan-server")

            notify(proc, "notifications/initialized")
            listed = call_rpc(proc, 2, "tools/list")
            names = {tool["name"] for tool in listed["result"]["tools"]}
            self.assertGreaterEqual(len(names), 5)
            self.assertIn("benchmark_plan", names)
            self.assertIn("apply_candidate", names)
            self.assertIn("local_neighborhood_search", names)
            self.assertIn("bounded_exhaustive_search", names)
            self.assertIn("export_candidate_fixture", names)

            ok = call_rpc(
                proc,
                3,
                "tools/call",
                {"name": "benchmark_plan", "arguments": {"fixture": FIXTURE_NAME, "modes": ["dev", "final"]}},
            )
            payload = tool_payload(ok)
            self.assertEqual(payload["results"]["dev"]["status"], "passed")
            self.assertEqual(payload["results"]["final"]["status"], "passed")
            self.assertIn("best_cycles", payload["results"]["dev"])

            bad = call_rpc(proc, 4, "tools/call", {"name": "missing_tool", "arguments": {}})
            self.assertTrue(bad["result"]["isError"])
            self.assertFalse(tool_payload(bad)["ok"])

    def test_neighborhood_diff_verified_apply_export_and_resume_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for name in ("benchmark.py", "verifier.py", "task.md", "kernel_plan.json"):
                shutil.copy2(ROOT / name, tmp / name)
            (tmp / ".harnessgym").mkdir(parents=True)
            shutil.copytree(ROOT / ".harnessgym" / "mcp", tmp / ".harnessgym" / "mcp")
            shutil.copytree(ROOT / ".harnessgym" / "fixtures", tmp / ".harnessgym" / "fixtures")
            temp_server = tmp / ".harnessgym" / "mcp" / "tensor-plan-server" / "tensor_plan_server.py"

            with ServerProcess(tmp, temp_server) as proc:
                call_rpc(proc, 1, "initialize")
                search = tool_payload(
                    call_rpc(
                        proc,
                        2,
                        "tools/call",
                        {
                            "name": "local_neighborhood_search",
                            "arguments": {
                                "fixture": FIXTURE_NAME,
                                "strategy": "checkpoint",
                                "max_evals": 500,
                                "keep": 6,
                                "include_final": True,
                                "record_history": True,
                            },
                        },
                    )
                )
                self.assertEqual(search["objective_metric"], "best_cycles")
                self.assertGreater(search["evaluated"], 0)
                self.assertLess(search["top"][0]["best_cycles"], search["seed"]["result"]["best_cycles"])
                self.assertEqual(search["top"][0]["final"]["status"], "passed")
                top_plan = search["top"][0]["plan"]

                diff = tool_payload(
                    call_rpc(
                        proc,
                        3,
                        "tools/call",
                        {
                            "name": "candidate_diff",
                            "arguments": {
                                "baseline_fixture": FIXTURE_NAME,
                                "candidate_plan": top_plan,
                                "modes": ["dev", "final"],
                            },
                        },
                    )
                )
                self.assertLess(diff["modes"]["dev"]["delta_best_cycles"], 0)
                self.assertIn("changed_fields", diff)

                verified = tool_payload(
                    call_rpc(
                        proc,
                        4,
                        "tools/call",
                        {
                            "name": "apply_best_verified",
                            "arguments": {
                                "plan": top_plan,
                                "dry_run": True,
                                "require_improvement": False,
                                "require_final_pass": True,
                            },
                        },
                    )
                )
                self.assertTrue(verified["ok"])
                self.assertFalse(verified["applied"])
                self.assertTrue(verified["numerical"]["ok"])

                exported = tool_payload(
                    call_rpc(
                        proc,
                        5,
                        "tools/call",
                        {
                            "name": "export_candidate_fixture",
                            "arguments": {
                                "name": "selftest_neighborhood_candidate",
                                "plan": top_plan,
                                "source": "self-test",
                                "notes": ["created in a temporary workspace"],
                                "overwrite": True,
                            },
                        },
                    )
                )
                self.assertTrue(exported["ok"])
                self.assertTrue((tmp / exported["path"]).exists())

                resume = tool_payload(call_rpc(proc, 6, "tools/call", {"name": "resume_search_history", "arguments": {"limit": 4}}))
                self.assertEqual(resume["objective_metric"], "best_cycles")
                self.assertGreaterEqual(len(resume["fixtures"]), 1)
                self.assertEqual(resume["recommended_next_tools"][0]["tool"], "local_neighborhood_search")

                bad = call_rpc(
                    proc,
                    7,
                    "tools/call",
                    {"name": "local_neighborhood_search", "arguments": {"fixture": FIXTURE_NAME, "strategy": "bad"}},
                )
                self.assertTrue(bad["result"]["isError"])
                self.assertFalse(tool_payload(bad)["ok"])

    def test_trace_search_validate_and_dry_run_apply_are_structured(self) -> None:
        with ServerProcess(ROOT, SERVER) as proc:
            call_rpc(proc, 1, "initialize")
            validate = tool_payload(call_rpc(proc, 2, "tools/call", {"name": "validate_plan", "arguments": {"fixture": FIXTURE_NAME}}))
            self.assertTrue(validate["ok"])
            self.assertEqual(validate["modes"]["dev"]["status"], "passed")

            trace = tool_payload(call_rpc(proc, 3, "tools/call", {"name": "trace_summary", "arguments": {"fixture": FIXTURE_NAME, "mode": "dev"}}))
            self.assertEqual(trace["summary"]["status"], "passed")
            self.assertGreater(len(trace["cases"]), 0)
            self.assertIn("throughput", trace["cases"][0])

            search = tool_payload(
                call_rpc(
                    proc,
                    4,
                    "tools/call",
                    {
                        "name": "search_plans",
                        "arguments": {"strategy": "quick", "max_evals": 5, "keep": 4, "record_history": False},
                    },
                )
            )
            self.assertEqual(search["objective_metric"], "best_cycles")
            self.assertTrue(any(row["name"].startswith("fixture:") for row in search["top"]))

            bounded = tool_payload(
                call_rpc(
                    proc,
                    5,
                    "tools/call",
                    {
                        "name": "bounded_exhaustive_search",
                        "arguments": {
                            "profile": "iteration3_dev_core",
                            "max_evals": 15,
                            "keep": 5,
                            "include_final": True,
                            "record_history": False,
                        },
                    },
                )
            )
            self.assertEqual(bounded["profile"], "iteration3_dev_core")
            self.assertEqual(bounded["evaluated"], 15)
            self.assertEqual(bounded["objective_metric"], "best_cycles")
            self.assertTrue(any(row["name"].startswith("fixture:") or row["name"] == "current" for row in bounded["top"]))

            apply = tool_payload(
                call_rpc(
                    proc,
                    6,
                    "tools/call",
                    {
                        "name": "apply_candidate",
                        "arguments": {
                            "fixture": FIXTURE_NAME,
                            "dry_run": True,
                            "require_improvement": False,
                            "require_final_pass": True,
                        },
                    },
                )
            )
            self.assertTrue(apply["ok"])
            self.assertFalse(apply["applied"])
            self.assertEqual(apply["final"]["status"], "passed")

            bad_profile = call_rpc(
                proc,
                7,
                "tools/call",
                {"name": "bounded_exhaustive_search", "arguments": {"profile": "not_a_profile"}},
            )
            self.assertTrue(bad_profile["result"]["isError"])
            self.assertFalse(tool_payload(bad_profile)["ok"])

    def test_apply_candidate_mutates_only_temp_workspace_and_keeps_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for name in ("benchmark.py", "verifier.py", "task.md", "kernel_plan.json"):
                shutil.copy2(ROOT / name, tmp / name)
            (tmp / ".harnessgym" / "mcp").mkdir(parents=True)
            (tmp / ".harnessgym" / "fixtures").mkdir(parents=True)
            shutil.copytree(ROOT / ".harnessgym" / "mcp" / "tensor-plan-server", tmp / ".harnessgym" / "mcp" / "tensor-plan-server")
            shutil.copy2(ROOT / ".harnessgym" / "fixtures" / f"{FIXTURE_NAME}.json", tmp / ".harnessgym" / "fixtures" / f"{FIXTURE_NAME}.json")
            shutil.copy2(ROOT / ".harnessgym" / "fixtures" / f"{ITER2_FIXTURE_NAME}.json", tmp / ".harnessgym" / "fixtures" / f"{ITER2_FIXTURE_NAME}.json")
            temp_server = tmp / ".harnessgym" / "mcp" / "tensor-plan-server" / "tensor_plan_server.py"

            before = json.loads((tmp / "kernel_plan.json").read_text(encoding="utf-8"))
            attempt1 = json.loads((tmp / ".harnessgym" / "fixtures" / f"{FIXTURE_NAME}.json").read_text(encoding="utf-8"))["plan"]
            fixture_to_apply = ITER2_FIXTURE_NAME if before == attempt1 else FIXTURE_NAME
            with ServerProcess(tmp, temp_server) as proc:
                call_rpc(proc, 1, "initialize")
                result = tool_payload(
                    call_rpc(
                        proc,
                        2,
                        "tools/call",
                        {
                            "name": "apply_candidate",
                            "arguments": {
                                "fixture": fixture_to_apply,
                                "dry_run": False,
                                "require_improvement": False,
                                "require_final_pass": True,
                            },
                        },
                    )
                )
            after = json.loads((tmp / "kernel_plan.json").read_text(encoding="utf-8"))
            self.assertTrue(result["applied"])
            self.assertNotEqual(before, after)
            self.assertTrue((tmp / result["backup_path"]).exists())
            self.assertEqual(after["q_layout"], "swizzled_mn")

    def test_numerical_tolerance_and_fixed_seed_properties(self) -> None:
        baseline = benchmark.load_plan(ROOT / "kernel_plan.json")
        for case in benchmark.CASES["dev"]:
            self.assertLessEqual(benchmark.estimate_error(baseline, case), benchmark.TOLERANCE)

        attempt1 = json.loads((ROOT / ".harnessgym" / "fixtures" / f"{FIXTURE_NAME}.json").read_text(encoding="utf-8"))["plan"]
        iteration2 = json.loads((ROOT / ".harnessgym" / "fixtures" / f"{ITER2_FIXTURE_NAME}.json").read_text(encoding="utf-8"))["plan"]
        self.assertEqual(benchmark.evaluate_plan(iteration2, "dev")["status"], "passed")
        self.assertEqual(benchmark.evaluate_plan(iteration2, "final")["status"], "passed")
        self.assertLess(benchmark.evaluate_plan(iteration2, "dev")["best_cycles"], benchmark.evaluate_plan(attempt1, "dev")["best_cycles"])

        failing = dict(baseline)
        failing.update({"softmax": "approx_poly", "accum": "scalar", "vector_width": 16})
        dim128_case = next(case for case in benchmark.CASES["final"] if case["dim"] == 128)
        self.assertGreater(benchmark.estimate_error(failing, dim128_case), benchmark.TOLERANCE)

        rng = random.Random(12345)
        valid_seen = 0
        keys = list(benchmark.VALID)
        while valid_seen < 60:
            plan = {key: rng.choice(benchmark.VALID[key]) for key in keys}
            errors = benchmark.validate_plan(plan)
            if errors:
                continue
            valid_seen += 1
            for mode in ("dev", "final"):
                result = benchmark.evaluate_plan(plan, mode)
                should_pass = result["max_abs"] <= benchmark.TOLERANCE
                self.assertEqual(result["status"] == "passed", should_pass)


if __name__ == "__main__":
    unittest.main(verbosity=2)
