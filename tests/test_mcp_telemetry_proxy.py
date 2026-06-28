import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any, BinaryIO


def write_msg(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    stream.flush()


def read_msg(stream: BinaryIO) -> dict[str, Any]:
    headers: dict[str, str] = {}
    line = stream.readline()
    while line not in {b"\r\n", b"\n", b""}:
        key, _, value = line.decode("ascii").partition(":")
        headers[key.lower()] = value.strip()
        line = stream.readline()
    length = int(headers["content-length"])
    return json.loads(stream.read(length).decode("utf-8"))


class McpTelemetryProxyTests(unittest.TestCase):
    def test_proxy_forwards_content_length_mcp_and_logs_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server = root / "server.py"
            server.write_text(
                textwrap.dedent(
                    r'''
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
                        return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


                    def write_msg(payload):
                        body = json.dumps(payload).encode("utf-8")
                        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
                        sys.stdout.buffer.flush()


                    while True:
                        msg = read_msg()
                        method = msg.get("method")
                        if method == "initialize":
                            write_msg({
                                "jsonrpc": "2.0",
                                "id": msg.get("id"),
                                "result": {
                                    "protocolVersion": "2024-11-05",
                                    "capabilities": {"tools": {}},
                                    "serverInfo": {"name": "fake", "version": "1.0"},
                                },
                            })
                        elif method == "notifications/initialized":
                            continue
                        elif method == "tools/list":
                            write_msg({
                                "jsonrpc": "2.0",
                                "id": msg.get("id"),
                                "result": {
                                    "tools": [
                                        {
                                            "name": "probe",
                                            "description": "Probe tool",
                                            "inputSchema": {"type": "object", "properties": {}},
                                        }
                                    ]
                                },
                            })
                        elif method == "tools/call":
                            write_msg({
                                "jsonrpc": "2.0",
                                "id": msg.get("id"),
                                "result": {
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": json.dumps({"ok": True, "args": msg.get("params", {}).get("arguments", {})}),
                                        }
                                    ]
                                },
                            })
                    '''
                ),
                encoding="utf-8",
            )
            proxy = Path(__file__).resolve().parents[1] / "src" / "harnessgym" / "mcp_telemetry_proxy.py"
            debug_path = root / ".harnessgym" / "mcp_proxy_debug.jsonl"
            env = os.environ.copy()
            env["HARNESSGYM_MCP_PROXY_DEBUG_PATH"] = str(debug_path)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(proxy),
                    "--server-name",
                    "fake-server",
                    "--response-timeout",
                    "5",
                    "--cwd",
                    str(root),
                    "--",
                    sys.executable,
                    str(server),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                write_msg(
                    process.stdin,
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05"},
                    },
                )
                initialize = read_msg(process.stdout)
                self.assertEqual(initialize["result"]["serverInfo"]["name"], "fake")

                write_msg(process.stdin, {"jsonrpc": "2.0", "method": "notifications/initialized"})
                write_msg(process.stdin, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
                tools = read_msg(process.stdout)
                self.assertEqual(tools["result"]["tools"][0]["name"], "probe")

                write_msg(
                    process.stdin,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "probe", "arguments": {"shape": [2, 4]}},
                    },
                )
                call = read_msg(process.stdout)
                self.assertEqual(call["result"]["content"][0]["type"], "text")

                telemetry_path = root / ".harnessgym" / "mcp_calls.jsonl"
                event = json.loads(telemetry_path.read_text(encoding="utf-8").strip())
                self.assertEqual(event["server"], "fake-server")
                self.assertEqual(event["method"], "tools/call")
                self.assertEqual(event["tool_name"], "probe")
                self.assertEqual(event["status"], "completed")
                self.assertEqual(event["argument_keys"], ["shape"])
                self.assertGreaterEqual(event["duration_ms"], 0)

                debug_events = [
                    json.loads(line)
                    for line in debug_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertIn("proxy_start", {event["event"] for event in debug_events})
                self.assertIn("client_request", {event["event"] for event in debug_events})
                self.assertIn("wrapped_response", {event["event"] for event in debug_events})
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
                process.terminate()
                process.wait(timeout=5)
