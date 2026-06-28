from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
        prog="harnessgym-mcp-telemetry-proxy",
        description="Proxy Content-Length framed MCP stdio and record HarnessGym tool-call telemetry.",
    )
    parser.add_argument("--cwd", default=None, help="Working directory for the wrapped MCP server.")
    parser.add_argument("--server-name", default=None, help="HarnessGym MCP server name for telemetry.")
    parser.add_argument(
        "--response-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for one wrapped MCP response.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Wrapped server command after --.")
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("harnessgym-mcp-telemetry-proxy: missing wrapped server command", file=sys.stderr)
        return 2

    return run_proxy(
        command,
        cwd=args.cwd,
        server_name=args.server_name,
        response_timeout_seconds=args.response_timeout,
    )


def run_proxy(
    command: list[str],
    cwd: str | None = None,
    server_name: str | None = None,
    response_timeout_seconds: float = 60.0,
) -> int:
    _log_debug_event(cwd, "proxy_start", server=server_name or "", command=command)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        while True:
            request = _read_content_length_message(sys.stdin.buffer, timeout_seconds=None)
            if request is None:
                _log_debug_event(cwd, "client_eof", server=server_name or "")
                return 0
            _log_debug_event(
                cwd,
                "client_request",
                server=server_name or "",
                method=str(request.get("method") or ""),
                request_id=request.get("id"),
            )
            tool_call = _tool_call_start(request, server_name=server_name)
            _write_content_length_message(process.stdin, request)
            if request.get("id") is None:
                _forward_available_stderr(process.stderr)
                continue

            response = _read_content_length_message(process.stdout, timeout_seconds=response_timeout_seconds)
            if response is None:
                stderr = _read_available_stderr(process.stderr)
                _log_debug_event(
                    cwd,
                    "wrapped_timeout",
                    server=server_name or "",
                    method=str(request.get("method") or ""),
                    request_id=request.get("id"),
                    stderr=stderr[-1000:],
                )
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32000,
                        "message": "wrapped MCP server did not produce a response",
                        "data": stderr,
                    },
                }
                if tool_call is not None:
                    tool_call["timed_out"] = True
            if tool_call is not None:
                _log_tool_call(tool_call, response, cwd=cwd)
            _log_debug_event(
                cwd,
                "wrapped_response",
                server=server_name or "",
                method=str(request.get("method") or ""),
                request_id=request.get("id"),
                has_error=bool(response.get("error")),
            )
            _write_content_length_message(sys.stdout.buffer, response)
            _forward_available_stderr(process.stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        _log_debug_event(cwd, "proxy_stop", server=server_name or "", return_code=process.poll())


def _tool_call_start(request: dict[str, Any], *, server_name: str | None) -> dict[str, Any] | None:
    if request.get("method") != "tools/call":
        return None
    params = request.get("params") if isinstance(request.get("params"), dict) else {}
    arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
    return {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "started_monotonic": time.monotonic(),
        "server": server_name or "",
        "request_id": request.get("id"),
        "method": "tools/call",
        "tool_name": str(params.get("name") or ""),
        "argument_keys": sorted(str(key) for key in arguments.keys()),
        "argument_json_bytes": _json_size(arguments),
        "timed_out": False,
    }


def _log_tool_call(call: dict[str, Any], response: dict[str, Any], *, cwd: str | None) -> None:
    duration_ms = max(0, int(round((time.monotonic() - float(call["started_monotonic"])) * 1000)))
    error = response.get("error") if isinstance(response.get("error"), dict) else None
    if call.get("timed_out"):
        status = "timeout"
    elif error:
        status = "error"
    else:
        status = "completed"
    event = {
        "created_at": call["created_at"],
        "server": call.get("server") or "",
        "request_id": call.get("request_id"),
        "method": call["method"],
        "tool_name": call.get("tool_name") or "",
        "duration_ms": duration_ms,
        "status": status,
        "error_message": str(error.get("message") or "") if error else "",
        "argument_keys": call.get("argument_keys") or [],
        "argument_json_bytes": call.get("argument_json_bytes") or 0,
        "result_json_bytes": _json_size(response.get("result") if not error else error),
    }
    log_path = _telemetry_path(cwd)
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError as exc:
        print(f"harnessgym MCP telemetry proxy could not write telemetry: {exc}", file=sys.stderr)


def _telemetry_path(cwd: str | None) -> Path | None:
    explicit = os.environ.get("HARNESSGYM_MCP_TELEMETRY_PATH")
    if explicit:
        return Path(explicit)
    workspace = os.environ.get("HARNESSGYM_WORKSPACE") or cwd
    if not workspace:
        return None
    return Path(workspace) / ".harnessgym" / "mcp_calls.jsonl"


def _debug_path(cwd: str | None) -> Path | None:
    explicit = os.environ.get("HARNESSGYM_MCP_PROXY_DEBUG_PATH")
    if explicit:
        return Path(explicit)
    if os.environ.get("HARNESSGYM_MCP_PROXY_DEBUG") not in {"1", "true", "yes"}:
        return None
    workspace = os.environ.get("HARNESSGYM_WORKSPACE") or cwd
    if not workspace:
        return None
    return Path(workspace) / ".harnessgym" / "mcp_proxy_debug.jsonl"


def _log_debug_event(cwd: str | None, event: str, **fields: Any) -> None:
    path = _debug_path(cwd)
    if path is None:
        return
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError as exc:
        print(f"harnessgym MCP telemetry proxy could not write debug log: {exc}", file=sys.stderr)


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _write_content_length_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    stream.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    stream.flush()


def _read_content_length_message(stream: BinaryIO, timeout_seconds: float | None) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    header_bytes = b""
    while b"\r\n\r\n" not in header_bytes and b"\n\n" not in header_bytes:
        if deadline is None:
            chunk = stream.read(1)
            if not chunk:
                return None
            header_bytes += chunk
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            return None
        chunk = os.read(stream.fileno(), 1)
        if not chunk:
            return None
        header_bytes += chunk
    separator = b"\r\n\r\n" if b"\r\n\r\n" in header_bytes else b"\n\n"
    header, _, extra = header_bytes.partition(separator)
    length = 0
    for line in header.decode("ascii", errors="replace").splitlines():
        name, _, value = line.partition(":")
        if name.lower() == "content-length":
            length = int(value.strip())
            break
    if length <= 0:
        return None
    body = extra
    while len(body) < length:
        if deadline is None:
            chunk = stream.read(length - len(body))
            if not chunk:
                return None
            body += chunk
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([stream], [], [], remaining)
        if not ready:
            return None
        body += os.read(stream.fileno(), length - len(body))
    return json.loads(body[:length].decode("utf-8"))


def _forward_available_stderr(stream: BinaryIO) -> None:
    text = _read_available_stderr(stream)
    if text:
        print(text, file=sys.stderr, end="" if text.endswith("\n") else "\n")


def _read_available_stderr(stream: BinaryIO) -> str:
    chunks: list[bytes] = []
    while True:
        ready, _, _ = select.select([stream], [], [], 0)
        if not ready:
            break
        chunk = os.read(stream.fileno(), 4096)
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks:
        return ""
    return b"".join(chunks).decode("utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
