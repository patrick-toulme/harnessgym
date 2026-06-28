from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MCP_CALL_LOG_NAME = "mcp_calls.jsonl"


def mcp_call_log_path(workspace: Path) -> Path:
    return workspace / ".harnessgym" / MCP_CALL_LOG_NAME


def read_mcp_call_events(workspace: Path) -> list[dict[str, Any]]:
    path = mcp_call_log_path(workspace)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summarize_mcp_call_events(workspace: Path, *, sample_limit: int = 20) -> dict[str, Any]:
    events = read_mcp_call_events(workspace)
    status_counts: dict[str, int] = {}
    called_tools: set[str] = set()
    servers: set[str] = set()
    successful_count = 0
    failed_count = 0
    for event in events:
        status = str(event.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "completed":
            successful_count += 1
        elif status in {"error", "failed", "timeout"}:
            failed_count += 1
        tool_name = event.get("tool_name")
        if tool_name:
            called_tools.add(str(tool_name))
        server = event.get("server")
        if server:
            servers.add(str(server))
    return {
        "path": str(mcp_call_log_path(workspace)),
        "count": len(events),
        "successful_count": successful_count,
        "failed_count": failed_count,
        "status_counts": dict(sorted(status_counts.items())),
        "called_tools": sorted(called_tools),
        "servers": sorted(servers),
        "samples": events[:sample_limit],
    }
