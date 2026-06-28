#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / ".harnessgym/mcp/h100_triton_rmsnorm/server.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("h100_triton_rmsnorm_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_fixture() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="h100-rmsnorm-mcp-test-"))
    (tmp / ".harnessgym/runs/run/iterations/1").mkdir(parents=True)
    (tmp / ".harnessgym/runs/run/baseline").mkdir(parents=True)
    (tmp / ".harnessgym/runs/run/checkpoints/best").mkdir(parents=True)
    write_json(tmp / ".harnessgym/activation.json", {"mcp_servers": [], "skills": []})
    write_json(tmp / ".harnessgym/runs/run/iterations/1/result.json", {"status": "running", "metrics": {}})
    (tmp / ".harnessgym/runs/run/baseline/baseline.stdout.txt").write_text(
        json.dumps({"status": "passed", "best_us": 150.0, "cases": []}) + "\n",
        encoding="utf-8",
    )
    (tmp / "benchmark.py").write_text(
        "CASES = {\n"
        "  'dev': [{'name': 'dev_r1024_d1024', 'rows': 1024, 'dim': 1024, 'seed': 1701}],\n"
        "  'final': [{'name': 'final_r192_d8192', 'rows': 192, 'dim': 8192, 'seed': 2704}],\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp / "kernel.py").write_text(
        "import triton.language as tl\n"
        "def f(gate):\n"
        "    return gate * tl.sigmoid(gate)\n",
        encoding="utf-8",
    )
    write_json(tmp / "kernel_config.json", {"num_warps": 1, "num_stages": 4, "block_size": 0})
    (tmp / ".harnessgym/runs/run/checkpoints/best/kernel.py").write_text(
        "import triton.language as tl\n"
        "def f(gate):\n"
        "    return gate * tl.sigmoid(gate)\n",
        encoding="utf-8",
    )
    write_json(
        tmp / ".harnessgym/runs/run/checkpoints/best/kernel_config.json",
        {"num_warps": 1, "num_stages": 4, "block_size": 0},
    )
    write_json(
        tmp / ".harnessgym/runs/run/checkpoints/best_manifest.json",
        {
            "checkpoint_path": str(tmp / ".harnessgym/runs/run/checkpoints/best"),
            "score": 103.328,
        },
    )
    (tmp / "verifier.py").write_text("", encoding="utf-8")
    (tmp / "remote_h100.py").write_text("", encoding="utf-8")
    return tmp


class FramedClient:
    def __init__(self, workspace: Path):
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER_PATH), "--workspace", str(workspace)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        self.proc.stdin.flush()
        return self.read_frame()

    def notify(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert self.proc.stdin is not None
        self.proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        self.proc.stdin.flush()

    def read_frame(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = self.proc.stdout.readline()
            self.assert_not_eof(line)
            if line in {b"\r\n", b"\n"}:
                break
            key, _, value = line.decode("ascii").partition(":")
            headers[key.lower()] = value.strip()
        length = int(headers["content-length"])
        body = self.proc.stdout.read(length)
        self.assert_not_eof(body)
        return json.loads(body.decode("utf-8"))

    @staticmethod
    def assert_not_eof(chunk: bytes) -> None:
        if chunk == b"":
            raise AssertionError("server closed stdout before response")


class H100TritonMcpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = make_fixture()
        self.client = FramedClient(self.fixture)

    def tearDown(self) -> None:
        self.client.close()
        shutil.rmtree(self.fixture, ignore_errors=True)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        self.assertIn("result", response)
        self.assertFalse(response["result"].get("isError"), response["result"])
        return json.loads(response["result"]["content"][0]["text"])

    def test_initialize_and_tools_list_are_content_length_framed(self) -> None:
        init = self.client.request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(init["result"]["serverInfo"]["name"], "h100-triton-rmsnorm-harness")
        self.client.notify({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        tools = self.client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {tool["name"] for tool in tools["result"]["tools"]}
        self.assertGreaterEqual(len(names), 5)
        self.assertIn("remote_health_check", names)
        self.assertIn("run_objective", names)
        self.assertIn("sweep_kernel_config", names)
        self.assertIn("sweep_launch_overrides", names)
        self.assertIn("sweep_silu_variants", names)
        self.assertIn("probe_silu_approximations", names)
        self.assertIn("sweep_silu_approximations", names)
        self.assertIn("joint_source_launch_search", names)
        self.assertIn("repeat_objective", names)
        self.assertIn("recommend_next_experiments", names)
        self.assertIn("guarded_final_verify", names)
        self.assertIn("restore_best_checkpoint", names)
        self.assertIn("numerical_probe", names)
        self.assertIn("diagnose_source", names)
        self.assertIn("rank_history", names)

    def test_inspect_context_and_numerical_probe_success_paths(self) -> None:
        context = self.call_tool("inspect_context", {})
        self.assertEqual(context["cases"]["dev"][0]["dim"], 1024)
        self.assertEqual(context["cases"]["final"][0]["dim"], 8192)
        self.assertTrue(context["source"]["kernel.py"]["exists"])
        probe = self.call_tool("numerical_probe", {"rows": 3, "dim": 7, "seed": 1234})
        self.assertEqual(probe["status"], "passed")
        self.assertLessEqual(probe["max_abs"], probe["tolerance"])
        self.assertEqual(probe["known_toy"]["rows"], 2)
        self.assertEqual(probe["fixed_seed_random"]["seed"], 1234)
        self.assertIn("exp2", probe["variants"])
        remote = self.call_tool("remote_health_check", {
            "dry_run": True,
            "host": "gpu.example",
            "port": "2222",
        })
        self.assertEqual(remote["status"], "dry_run")
        self.assertEqual(remote["mode"], "remote")
        self.assertEqual(remote["host"], "gpu.example")
        self.assertTrue(remote["command"])
        approx = self.call_tool("probe_silu_approximations", {
            "mode": "all",
            "variants": ["rational_m3n2"],
            "sample_rows": 2,
        })
        self.assertEqual(approx["variants"], ["rational_m3n2"])
        self.assertIn(approx["status"], {"passed", "failed"})
        self.assertEqual(approx["results"][0]["shape_proxy"]["mode"], "all")
        self.assertGreaterEqual(len(approx["results"][0]["shape_proxy"]["cases"]), 2)

    def test_guarded_verify_launch_silu_repeat_and_recommend_dry_run_paths(self) -> None:
        guarded = self.call_tool("guarded_final_verify", {"dry_run": True})
        self.assertEqual(guarded["status"], "dry_run")
        self.assertEqual(guarded["mode"], "final")
        self.assertEqual(guarded["threshold_score"], 103.328)
        launch = self.call_tool("sweep_launch_overrides", {"dry_run": True, "dims": [1024]})
        self.assertEqual(launch["status"], "dry_run")
        self.assertEqual(launch["mode"], "dev")
        self.assertGreaterEqual(launch["candidate_count"], 1)
        self.assertIn("rows_per_program_1024", launch["configs"][0])
        self.assertFalse(launch["source_supports_launch_overrides"])
        silu = self.call_tool("sweep_silu_variants", {
            "dry_run": True,
            "variants": ["exp", "sigmoid", "exp2"],
            "config_overlays": [{"num_warps_8192": 16, "num_stages_8192": 2}],
        })
        self.assertEqual(silu["status"], "dry_run")
        self.assertEqual(silu["candidate_count"], 3)
        self.assertIn("exp2", silu["silu_variants"])
        approx_sweep = self.call_tool("sweep_silu_approximations", {
            "dry_run": True,
            "variants": ["rational_m3n2"],
            "sample_rows": 2,
        })
        self.assertEqual(approx_sweep["status"], "dry_run")
        self.assertEqual(approx_sweep["candidate_count"], 1)
        self.assertIn("rational_m3n2", approx_sweep["approximation_variants"])
        self.assertEqual(approx_sweep["numerical_probes"][0]["variant"], "rational_m3n2")
        joint = self.call_tool("joint_source_launch_search", {
            "dry_run": True,
            "source_variants": ["current", "sigmoid", "rational_m3n2"],
            "config_overlays": [{}, {"num_warps_8192": 32, "num_stages_8192": 1}],
            "max_candidates": 8,
            "confirm_top_n": 1,
        })
        self.assertEqual(joint["status"], "dry_run")
        self.assertEqual(joint["candidate_count"], 6)
        self.assertEqual(joint["confirm_top_n"], 1)
        self.assertIn("rational_m3n2", joint["source_variants"])
        self.assertTrue(joint["objective"].startswith("minimize total best_us"))
        repeated = self.call_tool("repeat_objective", {"dry_run": True, "runs": 2})
        self.assertEqual(repeated["status"], "dry_run")
        self.assertEqual(repeated["mode"], "final")
        self.assertEqual(repeated["runs"], 2)
        recommended = self.call_tool("recommend_next_experiments", {})
        self.assertEqual(recommended["status"], "ready")
        self.assertTrue(recommended["shape_assumptions"]["final_is_held_out"])
        self.assertIn("recommended_commands", recommended)

    def test_restore_best_checkpoint_rolls_back_mutable_files(self) -> None:
        write_json(self.fixture / "kernel_config.json", {"num_warps": 8, "num_stages": 1, "block_size": 0})
        restored = self.call_tool("restore_best_checkpoint", {"files": ["kernel_config.json"]})
        self.assertEqual(restored["status"], "restored")
        self.assertEqual(restored["files"], ["kernel_config.json"])
        config = json.loads((self.fixture / "kernel_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["num_warps"], 1)

    def test_diagnose_source_and_update_result(self) -> None:
        diagnosis = self.call_tool("diagnose_source", {})
        self.assertTrue(diagnosis["patterns"]["uses_tl_sigmoid"])
        updated = self.call_tool("update_result_json", {
            "status": "tooling_built",
            "verified": True,
            "best_us": 103.328,
            "summary": "fixture update",
        })
        self.assertEqual(updated["metrics"]["score"], 103.328)
        result = json.loads(Path(updated["result_path"]).read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "tooling_built")
        self.assertTrue(result["verified"])

    def test_invalid_input_returns_tool_error(self) -> None:
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "run_objective", "arguments": {"mode": "not-a-mode"}},
        })
        self.assertTrue(response["result"].get("isError"))
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("mode must be one", payload["error"])
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 33,
            "method": "tools/call",
            "params": {"name": "remote_health_check", "arguments": {"timeout_seconds": 0}},
        })
        self.assertTrue(response["result"].get("isError"))
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("timeout_seconds", payload["error"])

    def test_sweep_validates_before_mutating_and_preserves_config_on_error(self) -> None:
        before = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "sweep_kernel_config",
                "arguments": {
                    "mode": "dev",
                    "configs": [{"num_warps": 3, "num_stages": 4, "block_size": 0}],
                },
            },
        })
        self.assertTrue(response["result"].get("isError"))
        after = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        self.assertEqual(before, after)

    def test_launch_sweep_invalid_input_preserves_config(self) -> None:
        before = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "sweep_launch_overrides",
                "arguments": {
                    "configs": [{"rows_per_program_1024": 3}],
                },
            },
        })
        self.assertTrue(response["result"].get("isError"))
        after = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        self.assertEqual(before, after)

    def test_silu_sweep_invalid_input_preserves_source_and_config(self) -> None:
        before_kernel = (self.fixture / "kernel.py").read_text(encoding="utf-8")
        before_config = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "sweep_silu_variants",
                "arguments": {
                    "variants": ["bad-variant"],
                },
            },
        })
        self.assertTrue(response["result"].get("isError"))
        after_kernel = (self.fixture / "kernel.py").read_text(encoding="utf-8")
        after_config = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        self.assertEqual(before_kernel, after_kernel)
        self.assertEqual(before_config, after_config)

    def test_silu_approximation_sweep_invalid_input_preserves_source_and_config(self) -> None:
        before_kernel = (self.fixture / "kernel.py").read_text(encoding="utf-8")
        before_config = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "sweep_silu_approximations",
                "arguments": {
                    "variants": ["bad-approx"],
                },
            },
        })
        self.assertTrue(response["result"].get("isError"))
        after_kernel = (self.fixture / "kernel.py").read_text(encoding="utf-8")
        after_config = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        self.assertEqual(before_kernel, after_kernel)
        self.assertEqual(before_config, after_config)

    def test_joint_search_invalid_input_preserves_source_and_config(self) -> None:
        before_kernel = (self.fixture / "kernel.py").read_text(encoding="utf-8")
        before_config = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        response = self.client.request({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "joint_source_launch_search",
                "arguments": {
                    "source_variants": ["bad-source"],
                    "config_overlays": [{}],
                },
            },
        })
        self.assertTrue(response["result"].get("isError"))
        after_kernel = (self.fixture / "kernel.py").read_text(encoding="utf-8")
        after_config = (self.fixture / "kernel_config.json").read_text(encoding="utf-8")
        self.assertEqual(before_kernel, after_kernel)
        self.assertEqual(before_config, after_config)


class DirectNumericalTests(unittest.TestCase):
    def test_numerical_probe_rejects_unbounded_shapes(self) -> None:
        server = load_server_module()
        tools = server.HarnessTools(ROOT)
        with self.assertRaises(server.ToolError):
            tools.numerical_probe({"rows": 4097, "dim": 4097})

    def test_silu_variant_renderer_supports_all_known_formulas(self) -> None:
        server = load_server_module()
        source = (
            "import triton.language as tl\n"
            "def f(gate):\n"
            "    return gate / (1.0 + tl.exp(-gate))\n"
        )
        for variant in ["exp", "sigmoid", "exp2"]:
            rendered, replacements = server.render_silu_variant(source, variant)
            self.assertEqual(replacements, 1)
            self.assertIn(server.SILU_VARIANTS[variant], rendered)

    def test_silu_approximation_probe_has_toy_and_shape_proxy_cases(self) -> None:
        server = load_server_module()
        fixture = make_fixture()
        try:
            result = server.HarnessTools(fixture).probe_silu_approximations({
                "mode": "all",
                "variants": ["rational_m3n2"],
                "sample_rows": 2,
            })
            self.assertEqual(result["variants"], ["rational_m3n2"])
            self.assertIn(result["status"], {"passed", "failed"})
            self.assertEqual(result["results"][0]["toy"]["rows"], 2)
            cases = result["results"][0]["shape_proxy"]["cases"]
            self.assertEqual({case["mode"] for case in cases}, {"dev", "final"})
        finally:
            shutil.rmtree(fixture, ignore_errors=True)

    def test_remote_health_check_classifies_success_and_ssh_failure(self) -> None:
        server = load_server_module()
        fixture = make_fixture()
        original_run_subprocess = server.run_subprocess

        def fake_success(command: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
            del cwd, timeout
            return {
                "command": command,
                "duration_seconds": 0.01,
                "return_code": 0,
                "stdout": "HG_REMOTE_OK\nHG_FREE_KB=1048576\nNVIDIA H100 80GB HBM3, 81920 MiB\n",
                "stderr": "",
            }

        def fake_refused(command: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
            del cwd, timeout
            return {
                "command": command,
                "duration_seconds": 0.01,
                "return_code": 255,
                "stdout": "",
                "stderr": "ssh: connect to host gpu.example port 2222: Connection refused\n",
            }

        try:
            server.run_subprocess = fake_success
            passed = server.HarnessTools(fixture).remote_health_check({
                "host": "gpu.example",
                "port": "2222",
                "min_free_mb": 256,
            })
            self.assertEqual(passed["status"], "passed")
            self.assertIsNone(passed["failure_stage"])
            self.assertEqual(passed["free_mb"], 1024.0)
            self.assertEqual(passed["gpu"], ["NVIDIA H100 80GB HBM3, 81920 MiB"])

            server.run_subprocess = fake_refused
            failed = server.HarnessTools(fixture).remote_health_check({
                "host": "gpu.example",
                "port": "2222",
            })
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["failure_stage"], "ssh")
            self.assertIn("Connection refused", failed["stderr_tail"])
        finally:
            server.run_subprocess = original_run_subprocess
            shutil.rmtree(fixture, ignore_errors=True)

    def test_silu_approximation_renderer_supports_exact_and_existing_approx_sources(self) -> None:
        server = load_server_module()
        exact_source = (
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def f(gate):\n"
            "    return gate * tl.sigmoid(gate)\n"
        )
        rendered, replacements = server.render_silu_approx_variant(exact_source, "rational_m3n2")
        self.assertEqual(replacements, 1)
        self.assertIn("_harnessgym_silu_approx(gate)", rendered)
        self.assertIn("import triton\n", rendered)
        approx_source = (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def _silu_rational_m3n2(gate):\n"
            "    return gate\n"
            "@triton.jit\n"
            "def f(gate):\n"
            "    return _silu_rational_m3n2(gate)\n"
        )
        rendered, replacements = server.render_silu_approx_variant(approx_source, "rational_m3n3")
        self.assertEqual(replacements, 1)
        self.assertIn("_harnessgym_silu_approx(gate)", rendered)

    def test_joint_source_renderer_converts_exact_and_approx_sources(self) -> None:
        server = load_server_module()
        exact_source = (
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def f(gate):\n"
            "    return gate * tl.sigmoid(gate)\n"
        )
        rendered, replacements = server.render_joint_source_variant(exact_source, "exp2")
        self.assertEqual(replacements, 1)
        self.assertIn("tl.exp2", rendered)
        approx_source = (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def _silu_rational_m3n2(gate):\n"
            "    return gate\n"
            "@triton.jit\n"
            "def f(gate):\n"
            "    return _silu_rational_m3n2(gate)\n"
        )
        rendered, replacements = server.render_joint_source_variant(approx_source, "sigmoid")
        self.assertEqual(replacements, 1)
        self.assertIn("gate * tl.sigmoid(gate)", rendered)
        self.assertNotIn("_harnessgym_silu_approx", rendered)

    def test_silu_sweep_rolls_back_after_fake_objective(self) -> None:
        server = load_server_module()
        fixture = make_fixture()

        class FakeTools(server.HarnessTools):
            def run_objective(self, args: dict[str, Any]) -> dict[str, Any]:
                text = (self.workspace / "kernel.py").read_text(encoding="utf-8")
                score = 99.0 if "tl.exp2" in text else 111.0
                return {
                    "status": "passed",
                    "return_code": 0,
                    "json": {
                        "status": "passed",
                        "best_us": score,
                        "cases": [
                            {"name": "final_r192_d8192", "dim": 8192, "rows": 192, "best_us": score}
                        ],
                    },
                }

        try:
            before_kernel = (fixture / "kernel.py").read_text(encoding="utf-8")
            before_config = (fixture / "kernel_config.json").read_text(encoding="utf-8")
            tools = FakeTools(fixture)
            result = tools.sweep_silu_variants({
                "mode": "final",
                "verifier": True,
                "variants": ["exp", "sigmoid", "exp2"],
                "config_overlays": [{"num_warps_8192": 16, "num_stages_8192": 2}],
            })
            self.assertEqual(result["best"]["variant"], "exp2")
            self.assertTrue(result["restored_original"])
            self.assertEqual((fixture / "kernel.py").read_text(encoding="utf-8"), before_kernel)
            self.assertEqual((fixture / "kernel_config.json").read_text(encoding="utf-8"), before_config)
        finally:
            shutil.rmtree(fixture, ignore_errors=True)

    def test_silu_approximation_sweep_rolls_back_after_fake_objective(self) -> None:
        server = load_server_module()
        fixture = make_fixture()

        class FakeTools(server.HarnessTools):
            def run_objective(self, args: dict[str, Any]) -> dict[str, Any]:
                text = (self.workspace / "kernel.py").read_text(encoding="utf-8")
                score = 97.0 if "0.00102942" in text else 120.0
                return {
                    "status": "passed",
                    "return_code": 0,
                    "json": {
                        "status": "passed",
                        "best_us": score,
                        "cases": [
                            {"name": "final_r192_d8192", "dim": 8192, "rows": 192, "best_us": score}
                        ],
                    },
                }

        try:
            before_kernel = (fixture / "kernel.py").read_text(encoding="utf-8")
            before_config = (fixture / "kernel_config.json").read_text(encoding="utf-8")
            result = FakeTools(fixture).sweep_silu_approximations({
                "mode": "final",
                "verifier": True,
                "variants": ["rational_m3n2"],
                "config_overlays": [{"num_warps_8192": 16, "num_stages_8192": 2}],
                "sample_rows": 2,
            })
            self.assertEqual(result["best"]["variant"], "rational_m3n2")
            self.assertTrue(result["restored_original"])
            self.assertEqual((fixture / "kernel.py").read_text(encoding="utf-8"), before_kernel)
            self.assertEqual((fixture / "kernel_config.json").read_text(encoding="utf-8"), before_config)
        finally:
            shutil.rmtree(fixture, ignore_errors=True)

    def test_joint_source_launch_search_rolls_back_and_confirms_fake_winner(self) -> None:
        server = load_server_module()
        fixture = make_fixture()

        class FakeTools(server.HarnessTools):
            def run_objective(self, args: dict[str, Any]) -> dict[str, Any]:
                del args
                text = (self.workspace / "kernel.py").read_text(encoding="utf-8")
                config = json.loads((self.workspace / "kernel_config.json").read_text(encoding="utf-8"))
                is_winner = "tl.exp2" in text and config.get("num_warps_8192") == 32
                score = 90.0 if is_winner else 120.0
                return {
                    "status": "passed",
                    "return_code": 0,
                    "json": {
                        "status": "passed",
                        "best_us": score,
                        "cases": [
                            {"name": "final_r192_d8192", "dim": 8192, "rows": 192, "best_us": score}
                        ],
                    },
                }

        try:
            before_kernel = (fixture / "kernel.py").read_text(encoding="utf-8")
            before_config = (fixture / "kernel_config.json").read_text(encoding="utf-8")
            result = FakeTools(fixture).joint_source_launch_search({
                "mode": "dev",
                "verifier": False,
                "source_variants": ["sigmoid", "exp2"],
                "config_overlays": [{}, {"num_warps_8192": 32, "num_stages_8192": 1}],
                "max_candidates": 4,
                "confirm_top_n": 1,
                "confirm_runs": 2,
            })
            self.assertEqual(result["initial_best"]["source_variant"], "exp2")
            self.assertEqual(result["best"]["source_variant"], "exp2")
            self.assertEqual(result["best"]["confirmation"]["median_score"], 90.0)
            self.assertEqual(len(result["confirmation_results"]), 1)
            self.assertTrue(result["restored_original"])
            self.assertEqual((fixture / "kernel.py").read_text(encoding="utf-8"), before_kernel)
            self.assertEqual((fixture / "kernel_config.json").read_text(encoding="utf-8"), before_config)
        finally:
            shutil.rmtree(fixture, ignore_errors=True)

    def test_repeat_objective_summarizes_scores_without_gpu(self) -> None:
        server = load_server_module()
        fixture = make_fixture()

        class FakeTools(server.HarnessTools):
            def __init__(self, workspace: Path):
                super().__init__(workspace)
                self.index = 0

            def run_objective(self, args: dict[str, Any]) -> dict[str, Any]:
                del args
                score = [105.0, 101.0, 103.0][self.index]
                self.index += 1
                return {
                    "status": "passed",
                    "return_code": 0,
                    "json": {
                        "status": "passed",
                        "best_us": score,
                        "cases": [
                            {"name": "final_r192_d8192", "dim": 8192, "rows": 192, "best_us": score}
                        ],
                    },
                }

        try:
            result = FakeTools(fixture).repeat_objective({"runs": 3, "mode": "final", "verifier": True})
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["min_score"], 101.0)
            self.assertEqual(result["median_score"], 103.0)
            self.assertEqual(result["max_score"], 105.0)
            self.assertEqual(result["cases"][0]["sample_count"], 3)
        finally:
            shutil.rmtree(fixture, ignore_errors=True)

    def test_manifest_contains_self_test_and_required_fields(self) -> None:
        manifest = json.loads((ROOT / ".harnessgym/mcp/h100_triton_rmsnorm/harnessgym-mcp.json").read_text(encoding="utf-8"))
        for key in ["name", "command", "args", "cwd", "enabled_tools", "timeouts", "self_test"]:
            self.assertIn(key, manifest)
        self.assertIn("run_objective", manifest["enabled_tools"])
        self.assertIn("sweep_kernel_config", manifest["enabled_tools"])
        self.assertIn("sweep_launch_overrides", manifest["enabled_tools"])
        self.assertIn("sweep_silu_variants", manifest["enabled_tools"])
        self.assertIn("probe_silu_approximations", manifest["enabled_tools"])
        self.assertIn("sweep_silu_approximations", manifest["enabled_tools"])
        self.assertIn("joint_source_launch_search", manifest["enabled_tools"])
        self.assertIn("repeat_objective", manifest["enabled_tools"])
        self.assertIn("recommend_next_experiments", manifest["enabled_tools"])
        self.assertIn("guarded_final_verify", manifest["enabled_tools"])
        self.assertEqual(manifest["self_test"]["command"][:2], ["python3", "server.py"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
