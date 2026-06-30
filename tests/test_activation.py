import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from harnessgym.activation import _replace_symlink, _smoke_mcp_server, activate_generated_harness
from harnessgym.models import Artifact, Registry


MCP_SERVER = r'''
import json
import sys


if "--self-test" in sys.argv:
    print(json.dumps({"status": "passed", "tests": ["content_length_smoke", "numerical_toy_case"]}))
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


class ActivationTests(unittest.TestCase):
    def test_activates_skill_symlink_and_project_mcp_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            skill_dir = workspace / ".harnessgym" / "skills" / "kernel-skill"
            skill_dir.mkdir(parents=True)
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                "---\nname: kernel-skill\ndescription: Use for kernel tasks.\n---\nDo the thing.\n",
                encoding="utf-8",
            )
            mcp_dir = workspace / ".harnessgym" / "mcp" / "kernel-tools"
            mcp_dir.mkdir(parents=True)
            (mcp_dir / "server.py").write_text(MCP_SERVER, encoding="utf-8")
            (mcp_dir / "mcp.json").write_text(
                json.dumps(
                    {
                        "name": "kernel-tools",
                        "command": "python3",
                        "args": [".harnessgym/mcp/kernel-tools/server.py"],
                        "cwd": ".",
                        "enabled_tools": ["run_verifier"],
                        "timeouts": {
                            "startup_seconds": 11,
                            "tool_seconds": 77,
                            "self_test_seconds": 33,
                        },
                        "self_test": {
                            "command": "python3",
                            "args": [".harnessgym/mcp/kernel-tools/server.py", "--self-test"],
                            "timeout_seconds": 22,
                        },
                    }
                ),
                encoding="utf-8",
            )
            registry = Registry(
                artifacts=[
                    Artifact(
                        id="skill:.harnessgym/skills/kernel-skill/SKILL.md",
                        kind="skill",
                        path=".harnessgym/skills/kernel-skill/SKILL.md",
                    ),
                    Artifact(
                        id="skill:.harnessgym/skills/kernel-skill",
                        kind="skill",
                        path=".harnessgym/skills/kernel-skill",
                    ),
                    Artifact(
                        id="mcp:.harnessgym/mcp/kernel-tools/mcp.json",
                        kind="mcp",
                        path=".harnessgym/mcp/kernel-tools/mcp.json",
                    ),
                    Artifact(
                        id="mcp:.harnessgym/mcp/kernel-tools/server.py",
                        kind="mcp",
                        path=".harnessgym/mcp/kernel-tools/server.py",
                    ),
                ]
            )

            report = activate_generated_harness(workspace, registry)

            self.assertEqual(len(report["skills"]), 1)
            self.assertEqual(len(report["mcp_servers"]), 1)
            self.assertTrue(report["mcp_servers"][0]["active"])
            self.assertEqual(report["mcp_servers"][0]["smoke"]["status"], "passed")
            self.assertEqual(report["mcp_servers"][0]["self_test"]["status"], "passed")
            self.assertEqual(report["quality_gate"]["status"], "passed")
            self.assertEqual(report["mcp_servers"][0]["startup_timeout_sec"], 11)
            self.assertEqual(report["mcp_servers"][0]["tool_timeout_sec"], 77)
            self.assertEqual(report["mcp_servers"][0]["self_test_timeout_sec"], 33)
            self.assertEqual(report["mcp_call_helper"]["relative_path"], ".harnessgym/runtime/mcp_call.py")
            self.assertTrue((workspace / ".harnessgym" / "runtime" / "mcp_call.py").exists())
            activated_skill = workspace / ".agents" / "skills" / "kernel-skill"
            claude_activated_skill = workspace / ".claude" / "skills" / "kernel-skill"
            self.assertTrue(activated_skill.is_symlink())
            self.assertTrue(claude_activated_skill.is_symlink())
            self.assertEqual(report["skills"][0]["codex_activated_path"], str(activated_skill))
            self.assertEqual(report["skills"][0]["claude_activated_path"], str(claude_activated_skill))
            config = (workspace / ".codex" / "config.toml").read_text(encoding="utf-8")
            self.assertIn("[mcp_servers.kernel-tools]", config)
            self.assertIn(f"command = {json.dumps(sys.executable)}", config)
            self.assertIn("mcp_telemetry_proxy.py", config)
            self.assertIn('"--server-name"', config)
            self.assertIn('"kernel-tools"', config)
            self.assertIn('"--response-timeout"', config)
            self.assertIn('"77"', config)
            self.assertIn('"python3"', config)
            self.assertIn('".harnessgym/mcp/kernel-tools/server.py"', config)
            self.assertIn("tool_timeout_sec = 82", config)
            self.assertIn('enabled_tools = ["run_verifier"]', config)

    def test_rewrites_copied_repo_local_mcp_absolute_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            mcp_dir = workspace / ".harnessgym" / "mcp" / "kernel-tools"
            mcp_dir.mkdir(parents=True)
            (mcp_dir / "server.py").write_text(MCP_SERVER, encoding="utf-8")
            (mcp_dir / "mcp.json").write_text(
                json.dumps(
                    {
                        "name": "kernel-tools",
                        "command": "python3",
                        "args": [".harnessgym/mcp/kernel-tools/server.py"],
                        "cwd": "/old/training/workspace",
                    }
                ),
                encoding="utf-8",
            )
            registry = Registry(
                artifacts=[
                    Artifact(
                        id="mcp:.harnessgym/mcp/kernel-tools/mcp.json",
                        kind="mcp",
                        path=".harnessgym/mcp/kernel-tools/mcp.json",
                    )
                ]
            )

            report = activate_generated_harness(workspace, registry)

            self.assertEqual(report["mcp_servers"][0]["cwd"], str(workspace))
            self.assertFalse(report["mcp_servers"][0]["active"])
            self.assertEqual(report["mcp_servers"][0]["quality_gate"]["status"], "failed")
            self.assertEqual(report["quality_gate"]["inactive_mcp_count"], 1)
            self.assertFalse((workspace / ".codex" / "config.toml").exists())
            self.assertEqual(report["mcp_servers"][0]["self_test"]["status"], "not_configured")

    def test_rewrites_self_test_cwd_and_parses_millisecond_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            old_workspace = root / "old-workspace"
            old_workspace.mkdir()
            mcp_dir = workspace / ".harnessgym" / "mcp" / "kernel-tools"
            mcp_dir.mkdir(parents=True)
            (mcp_dir / "server.py").write_text(MCP_SERVER, encoding="utf-8")
            (mcp_dir / "mcp.json").write_text(
                json.dumps(
                    {
                        "name": "kernel-tools",
                        "command": "python3",
                        "args": [".harnessgym/mcp/kernel-tools/server.py"],
                        "cwd": str(old_workspace),
                        "timeouts": {
                            "initialize_ms": 5000,
                            "tool_call_ms": 120000,
                            "self_test_ms": 45000,
                        },
                        "self_test": {
                            "command": "python3",
                            "args": [".harnessgym/mcp/kernel-tools/server.py", "--self-test"],
                            "cwd": str(old_workspace),
                            "timeout_ms": 120000,
                        },
                    }
                ),
                encoding="utf-8",
            )
            registry = Registry(
                artifacts=[
                    Artifact(
                        id="mcp:.harnessgym/mcp/kernel-tools/mcp.json",
                        kind="mcp",
                        path=".harnessgym/mcp/kernel-tools/mcp.json",
                    )
                ]
            )

            report = activate_generated_harness(workspace, registry)

            server = report["mcp_servers"][0]
            self.assertTrue(server["active"])
            self.assertEqual(server["startup_timeout_sec"], 5)
            self.assertEqual(server["tool_timeout_sec"], 120)
            self.assertEqual(server["self_test_timeout_sec"], 45)
            self.assertEqual(server["self_test"]["status"], "passed")
            config = (workspace / ".codex" / "config.toml").read_text(encoding="utf-8")
            self.assertIn(f'cwd = "{workspace}"', config)

    def test_replace_symlink_preserves_preexisting_real_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            link_path = root / "skill_link"
            link_path.mkdir()
            (link_path / "OLD.txt").write_text("old", encoding="utf-8")
            target = root / "new_target"
            target.mkdir()
            (target / "NEW.txt").write_text("new", encoding="utf-8")

            _replace_symlink(link_path, target)

            self.assertFalse(link_path.is_symlink())
            self.assertTrue((link_path / "OLD.txt").exists())
            self.assertFalse((link_path / "NEW.txt").exists())

    def test_replace_symlink_replaces_stale_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            link_path = root / "skill_link"
            link_path.write_text("old", encoding="utf-8")
            target = root / "new_target"
            target.mkdir()
            (target / "NEW.txt").write_text("new", encoding="utf-8")

            _replace_symlink(link_path, target)

            self.assertTrue(link_path.is_symlink())
            self.assertEqual(Path(os.readlink(link_path)), target)
            self.assertTrue((link_path / "NEW.txt").exists())

    def test_smoke_mcp_server_uses_configured_startup_timeout(self) -> None:
        # A slow server that sleeps past the default 3s but within a configured
        # startup_timeout_sec should pass smoke when the configured timeout is used.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slow_server = root / "slow_server.py"
            slow_server.write_text(
                "\n".join(
                    [
                        "import json, sys, time",
                        "def read_msg():",
                        "    headers = {}",
                        "    while True:",
                        "        line = sys.stdin.buffer.readline()",
                        "        if line in (b'\\r\\n', b'\\n', b''): break",
                        "        k,_,v = line.decode('ascii').partition(':')",
                        "        headers[k.lower()] = v.strip()",
                        "    length = int(headers.get('content-length','0'))",
                        "    return json.loads(sys.stdin.buffer.read(length).decode('utf-8'))",
                        "def write_msg(p):",
                        "    body = json.dumps(p).encode('utf-8')",
                        "    sys.stdout.buffer.write(f'Content-Length: {len(body)}\\r\\n\\r\\n'.encode('ascii') + body)",
                        "    sys.stdout.buffer.flush()",
                        "time.sleep(2)  # slow startup",
                        "while True:",
                        "    msg = read_msg()",
                        "    if msg is None: break",
                        "    if msg.get('method') == 'initialize':",
                        "        write_msg({'jsonrpc':'2.0','id':msg.get('id'),'result':{'protocolVersion':'2024-11-05','capabilities':{'tools':{}},'serverInfo':{'name':'slow','version':'1.0'}}})",
                        "    elif msg.get('method') == 'tools/list':",
                        "        write_msg({'jsonrpc':'2.0','id':msg.get('id'),'result':{'tools':[{'name':'probe','description':'p','inputSchema':{'type':'object','properties':{}}}]}})",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            server = {
                "name": "slow",
                "command": sys.executable,
                "args": [str(slow_server)],
                "cwd": str(root),
                "startup_timeout_sec": 10,
            }
            # With the configured timeout (10s), the 2s-sleep server should pass.
            result = _smoke_mcp_server(server, timeout_seconds=float(server["startup_timeout_sec"]))
            self.assertEqual(result["status"], "passed")
            # With the old hardcoded 3s default, it would also pass here (2s sleep),
            # but the point is the configured value is now respected.
