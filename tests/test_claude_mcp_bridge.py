import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class ClaudeMcpBridgeTests(unittest.TestCase):
    def test_bridge_translates_newline_json_to_content_length_server(self) -> None:
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
            bridge = Path(__file__).resolve().parents[1] / "src" / "harnessgym" / "claude_mcp_bridge.py"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(bridge),
                    "--server-name",
                    "fake-server",
                    "--cwd",
                    str(root),
                    "--",
                    sys.executable,
                    str(server),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                process.stdin.write(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 0,
                            "method": "initialize",
                            "params": {"protocolVersion": "2025-11-25"},
                        }
                    )
                    + "\n"
                )
                process.stdin.flush()
                initialize = json.loads(process.stdout.readline())
                self.assertEqual(initialize["result"]["protocolVersion"], "2025-11-25")

                process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
                process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n")
                process.stdin.flush()
                tools = json.loads(process.stdout.readline())
                self.assertEqual(tools["result"]["tools"][0]["name"], "probe")

                process.stdin.write(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 2,
                            "method": "tools/call",
                            "params": {"name": "probe", "arguments": {"shape": [2, 4]}},
                        }
                    )
                    + "\n"
                )
                process.stdin.flush()
                call = json.loads(process.stdout.readline())
                self.assertEqual(call["result"]["content"][0]["type"], "text")

                telemetry_path = root / ".harnessgym" / "mcp_calls.jsonl"
                event = json.loads(telemetry_path.read_text(encoding="utf-8").strip())
                self.assertEqual(event["server"], "fake-server")
                self.assertEqual(event["method"], "tools/call")
                self.assertEqual(event["tool_name"], "probe")
                self.assertEqual(event["status"], "completed")
                self.assertEqual(event["argument_keys"], ["shape"])
                self.assertGreaterEqual(event["duration_ms"], 0)
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
                process.terminate()
                process.wait(timeout=5)
