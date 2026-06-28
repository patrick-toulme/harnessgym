from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnessgym-mcp-call",
        description="Call an activated HarnessGym MCP server through the telemetry proxy.",
    )
    parser.add_argument("--workspace", default=".", help="Workspace containing .harnessgym/.")
    parser.add_argument("--server", required=True, help="Activated MCP server name.")
    parser.add_argument("--tool", help="Tool name to call.")
    parser.add_argument("--arguments", default="{}", help="JSON object passed as tool arguments.")
    parser.add_argument("--arguments-file", help="Path to a JSON file passed as tool arguments.")
    parser.add_argument("--list-tools", action="store_true", help="List server tools instead of calling a tool.")
    parser.add_argument("--raw", action="store_true", help="Print the raw JSON-RPC response.")
    parser.add_argument("--timeout", type=float, default=None, help="Override the server tool timeout in seconds.")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    if not args.list_tools and not args.tool:
        parser.error("--tool is required unless --list-tools is passed")
    try:
        arguments = _load_arguments(args.arguments, args.arguments_file)
        server = _load_server(workspace, args.server)
        response = call_server(
            workspace=workspace,
            server=server,
            tool_name=args.tool,
            arguments=arguments,
            list_tools=bool(args.list_tools),
            timeout_seconds=args.timeout,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(response, indent=2, sort_keys=True))
        return 0 if "error" not in response else 1

    payload = _payload_from_response(response)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if "error" in response:
        return 1
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    return 1 if result.get("isError") else 0


def call_server(
    *,
    workspace: Path,
    server: dict[str, Any],
    tool_name: str | None,
    arguments: dict[str, Any] | None = None,
    list_tools: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    tool_timeout = float(timeout_seconds if timeout_seconds is not None else server.get("tool_timeout_sec", 60))
    command = _proxy_command(server, tool_timeout)
    env = os.environ.copy()
    env.setdefault("HARNESSGYM_WORKSPACE", str(workspace))
    process = subprocess.Popen(
        command,
        cwd=str(server["cwd"]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        _write_message(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"clientInfo": {"name": "harnessgym-mcp-call", "version": "1"}},
            },
        )
        initialize = _read_message(process.stdout, timeout_seconds=min(max(tool_timeout, 1.0), 30.0))
        if initialize is None or "error" in initialize:
            raise RuntimeError(f"MCP initialize failed: {initialize!r}; stderr={_read_available(process.stderr)}")
        _write_message(process.stdin, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        if list_tools:
            _write_message(process.stdin, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            response = _read_message(process.stdout, timeout_seconds=min(max(tool_timeout, 1.0), 30.0))
        else:
            _write_message(
                process.stdin,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments or {}},
                },
            )
            response = _read_message(process.stdout, timeout_seconds=tool_timeout)
        if response is None:
            raise RuntimeError(f"MCP call timed out; stderr={_read_available(process.stderr)}")
        return response
    finally:
        _terminate(process)


def _proxy_command(server: dict[str, Any], tool_timeout: float) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().parent / "mcp_telemetry_proxy.py"),
        "--server-name",
        str(server["name"]),
        "--response-timeout",
        str(tool_timeout),
        "--cwd",
        str(server["cwd"]),
        "--",
        str(server["command"]),
        *[str(arg) for arg in server.get("args", [])],
    ]


def _load_server(workspace: Path, name: str) -> dict[str, Any]:
    activation_path = workspace / ".harnessgym" / "activation.json"
    if activation_path.exists():
        activation = json.loads(activation_path.read_text(encoding="utf-8"))
        for server in activation.get("mcp_servers", []):
            if isinstance(server, dict) and server.get("name") == name and server.get("active") is not False:
                loaded = dict(server)
                loaded["cwd"] = str(Path(str(loaded.get("cwd") or workspace)).resolve())
                return loaded
    manifest = _find_manifest(workspace, name)
    if manifest is None:
        raise FileNotFoundError(f"could not find active MCP server {name!r}")
    loaded_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    cwd = loaded_manifest.get("cwd") or "."
    cwd_path = Path(str(cwd))
    if not cwd_path.is_absolute():
        cwd_path = (workspace / cwd_path).resolve()
    timeouts = loaded_manifest.get("timeouts") if isinstance(loaded_manifest.get("timeouts"), dict) else {}
    return {
        "name": str(loaded_manifest.get("name") or manifest.parent.name),
        "command": str(loaded_manifest["command"]),
        "args": [str(arg) for arg in loaded_manifest.get("args", [])],
        "cwd": str(cwd_path),
        "tool_timeout_sec": int(
            loaded_manifest.get("tool_timeout_sec")
            or timeouts.get("tool_seconds")
            or timeouts.get("tool_timeout_sec")
            or 60
        ),
    }


def _find_manifest(workspace: Path, name: str) -> Path | None:
    root = workspace / ".harnessgym" / "mcp"
    if not root.exists():
        return None
    for candidate in root.glob("**/*.json"):
        if candidate.name not in {"mcp.json", "server.json", "harnessgym-mcp.json"}:
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("name") == name or candidate.parent.name == name:
            return candidate
    return None


def _load_arguments(argument_json: str, argument_file: str | None) -> dict[str, Any]:
    if argument_file:
        value = json.loads(Path(argument_file).read_text(encoding="utf-8"))
    else:
        value = json.loads(argument_json)
    if not isinstance(value, dict):
        raise ValueError("MCP tool arguments must be a JSON object")
    return value


def _payload_from_response(response: dict[str, Any]) -> Any:
    if "error" in response:
        return response
    result = response.get("result")
    if not isinstance(result, dict):
        return response
    content = result.get("content")
    if isinstance(content, list) and content and isinstance(content[0], dict):
        text = content[0].get("text")
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text, "isError": bool(result.get("isError"))}
    return result


def _write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    stream.flush()


def _read_message(stream: BinaryIO, timeout_seconds: float) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    headers: dict[str, str] = {}
    line = _readline(stream, deadline)
    if line is None:
        return None
    while line not in {b"\r\n", b"\n", b""}:
        text = line.decode("ascii", errors="replace").strip()
        if ":" in text:
            key, value = text.split(":", 1)
            headers[key.lower()] = value.strip()
        line = _readline(stream, deadline)
        if line is None:
            return None
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = _read_exact(stream, length, deadline)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


def _readline(stream: BinaryIO, deadline: float) -> bytes | None:
    line = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            return None
        chunk = os.read(stream.fileno(), 1)
        if not chunk:
            return bytes(line) if line else None
        line.extend(chunk)
        if chunk == b"\n":
            return bytes(line)


def _read_exact(stream: BinaryIO, length: int, deadline: float) -> bytes | None:
    chunks: list[bytes] = []
    remaining_length = length
    while remaining_length > 0:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            return None
        chunk = os.read(stream.fileno(), remaining_length)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining_length -= len(chunk)
    return b"".join(chunks)


def _read_available(stream: BinaryIO) -> str:
    chunks: list[bytes] = []
    while True:
        ready, _, _ = select.select([stream], [], [], 0)
        if not ready:
            break
        chunk = os.read(stream.fileno(), 4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
