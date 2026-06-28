import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SERVER = r'''
import json
import sys


def read_msg():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.lower()] = value.strip()
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
        write_msg({
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "probe-server", "version": "1.0"},
            },
        })
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        write_msg({
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {"tools": [{"name": "probe", "description": "Probe", "inputSchema": {"type": "object"}}]},
        })
    elif method == "tools/call":
        params = msg.get("params", {})
        write_msg({
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "result": {
                "content": [{"type": "text", "text": json.dumps({"ok": True, "arguments": params.get("arguments", {})})}],
                "isError": False,
            },
        })
'''


class McpClientTests(unittest.TestCase):
    def test_cli_calls_activated_server_through_telemetry_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            server_path = workspace / "server.py"
            server_path.write_text(textwrap.dedent(SERVER), encoding="utf-8")
            harness_dir = workspace / ".harnessgym"
            harness_dir.mkdir()
            (harness_dir / "activation.json").write_text(
                json.dumps(
                    {
                        "mcp_servers": [
                            {
                                "name": "probe-server",
                                "active": True,
                                "command": sys.executable,
                                "args": [str(server_path)],
                                "cwd": str(workspace),
                                "tool_timeout_sec": 5,
                                "enabled_tools": ["probe"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "harnessgym.mcp_client",
                    "--workspace",
                    str(workspace),
                    "--server",
                    "probe-server",
                    "--tool",
                    "probe",
                    "--arguments",
                    '{"shape":[2,4]}',
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"ok": True, "arguments": {"shape": [2, 4]}})
            telemetry = json.loads((harness_dir / "mcp_calls.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(telemetry["server"], "probe-server")
            self.assertEqual(telemetry["tool_name"], "probe")
            self.assertEqual(telemetry["status"], "completed")


if __name__ == "__main__":
    unittest.main()
