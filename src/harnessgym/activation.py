from __future__ import annotations

import json
import os
import re
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .models import Registry
from .registry import active_artifacts, artifact_is_quarantined


BEGIN_MARKER = "# BEGIN HARNESSGYM MCP"
END_MARKER = "# END HARNESSGYM MCP"


def activate_generated_harness(workspace: Path, registry: Registry) -> dict[str, Any]:
    """Expose generated repo-local HarnessGym artifacts through runner-native paths."""
    report: dict[str, Any] = {
        "skills": [],
        "mcp_servers": [],
        "quarantined_artifacts": [
            {"kind": artifact.kind, "path": artifact.path}
            for artifact in registry.artifacts
            if artifact_is_quarantined(artifact)
        ],
        "warnings": [],
        "quality_gate": {
            "status": "passed",
            "active_mcp_count": 0,
            "inactive_mcp_count": 0,
            "active_tool_count": 0,
            "warnings": [],
        },
    }
    _activate_skills(workspace, registry, report)
    _activate_mcp_servers(workspace, registry, report)
    _finalize_quality_gate(report)
    activation_path = workspace / ".harnessgym" / "activation.json"
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    activation_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _activate_skills(workspace: Path, registry: Registry, report: dict[str, Any]) -> None:
    codex_skills_root = workspace / ".agents" / "skills"
    claude_skills_root = workspace / ".claude" / "skills"
    seen_targets: set[Path] = set()
    for artifact in registry.artifacts:
        if artifact_is_quarantined(artifact):
            continue
        if artifact.kind != "skill":
            continue
        target = _skill_target(workspace, artifact.path)
        if target is None:
            report["warnings"].append(f"skill artifact has no SKILL.md: {artifact.path}")
            continue
        if target in seen_targets:
            continue
        seen_targets.add(target)
        codex_skills_root.mkdir(parents=True, exist_ok=True)
        claude_skills_root.mkdir(parents=True, exist_ok=True)
        codex_link_path = codex_skills_root / target.parent.name
        claude_link_path = claude_skills_root / target.parent.name
        _replace_symlink(codex_link_path, target.parent)
        _replace_symlink(claude_link_path, target.parent)
        report["skills"].append(
            {
                "artifact_path": artifact.path,
                "skill_path": str(target),
                "activated_path": str(codex_link_path),
                "codex_activated_path": str(codex_link_path),
                "claude_activated_path": str(claude_link_path),
            }
        )


def _skill_target(workspace: Path, path: str) -> Path | None:
    candidate = workspace / path
    if candidate.is_file() and candidate.name == "SKILL.md":
        return candidate
    if candidate.is_dir() and (candidate / "SKILL.md").exists():
        return candidate / "SKILL.md"
    if candidate.is_file():
        sibling = candidate.parent / "SKILL.md"
        if sibling.exists():
            return sibling
    return None


def _replace_symlink(link_path: Path, target: Path) -> None:
    if link_path.is_symlink():
        if Path(os.readlink(link_path)) == target:
            return
        link_path.unlink()
    elif link_path.exists():
        return
    os.symlink(target, link_path, target_is_directory=target.is_dir())


def _activate_mcp_servers(workspace: Path, registry: Registry, report: dict[str, Any]) -> None:
    servers_by_name: dict[str, dict[str, Any]] = {}
    for artifact in active_artifacts(registry):
        if artifact.kind != "mcp":
            continue
        manifest_path = _mcp_manifest_path(workspace, artifact.path)
        if manifest_path is None:
            report["warnings"].append(f"mcp artifact has no manifest: {artifact.path}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            server = _normalize_mcp_manifest(workspace, manifest_path, manifest)
        except Exception as exc:
            report["warnings"].append(f"invalid mcp manifest {artifact.path}: {exc}")
            continue
        if server["name"] in servers_by_name:
            continue
        server["smoke"] = _smoke_mcp_server(server)
        if server["smoke"].get("status") != "passed":
            report["warnings"].append(f"mcp smoke failed for {server['name']}: {server['smoke'].get('message')}")
        server["self_test"] = _run_mcp_self_test(server)
        if server["self_test"].get("status") in {"failed", "timeout"}:
            report["warnings"].append(
                f"mcp self-test failed for {server['name']}: {server['self_test'].get('message')}"
            )
        server["quality_gate"] = _mcp_quality_gate(server)
        server["active"] = server["quality_gate"]["status"] == "passed"
        if not server["active"]:
            report["warnings"].append(f"mcp quality gate failed for {server['name']}: {server['quality_gate']['message']}")
        else:
            servers_by_name[server["name"]] = server
        report["mcp_servers"].append({"artifact_path": artifact.path, **server})
    _write_project_codex_config(workspace, list(servers_by_name.values()))
    if servers_by_name:
        report["mcp_call_helper"] = _write_mcp_call_helper(workspace)


def _mcp_quality_gate(server: dict[str, Any]) -> dict[str, Any]:
    smoke_status = server.get("smoke", {}).get("status") if isinstance(server.get("smoke"), dict) else "failed"
    self_test_status = (
        server.get("self_test", {}).get("status") if isinstance(server.get("self_test"), dict) else "not_configured"
    )
    smoke_tools = []
    if isinstance(server.get("smoke"), dict):
        smoke_tools = [str(tool) for tool in server["smoke"].get("tools", []) if tool]
    manifest_tools = [str(tool) for tool in server.get("enabled_tools", []) if tool]
    checks = {
        "smoke": smoke_status == "passed",
        "self_test": self_test_status == "passed",
        "tool_inventory": bool(smoke_tools or manifest_tools),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "message": "passed" if not failed else "failed checks: " + ", ".join(failed),
        "smoke_status": smoke_status,
        "self_test_status": self_test_status,
        "smoke_tools": smoke_tools,
        "manifest_tools": manifest_tools,
    }


def _finalize_quality_gate(report: dict[str, Any]) -> None:
    active_servers = [server for server in report["mcp_servers"] if server.get("active")]
    inactive_servers = [server for server in report["mcp_servers"] if not server.get("active")]
    active_tools: set[str] = set()
    for server in active_servers:
        active_tools.update(str(tool) for tool in server.get("enabled_tools", []) if tool)
        smoke = server.get("smoke")
        if isinstance(smoke, dict):
            active_tools.update(str(tool) for tool in smoke.get("tools", []) if tool)
    warnings = [str(warning) for warning in report.get("warnings", [])]
    report["quality_gate"] = {
        "status": "passed" if not inactive_servers and not warnings else "failed",
        "active_mcp_count": len(active_servers),
        "inactive_mcp_count": len(inactive_servers),
        "active_tool_count": len(active_tools),
        "warnings": warnings,
    }


def _mcp_manifest_path(workspace: Path, path: str) -> Path | None:
    candidate = workspace / path
    if candidate.is_file() and candidate.suffix == ".json":
        return candidate
    if candidate.is_dir():
        for name in ("mcp.json", "server.json", "harnessgym-mcp.json"):
            manifest = candidate / name
            if manifest.exists():
                return manifest
    if candidate.is_file():
        for name in ("mcp.json", "server.json", "harnessgym-mcp.json"):
            manifest = candidate.parent / name
            if manifest.exists():
                return manifest
    return None


def _normalize_mcp_manifest(workspace: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    name = _safe_server_name(str(manifest.get("name") or manifest_path.parent.name))
    command = str(manifest["command"])
    args = [str(arg) for arg in manifest.get("args", [])]
    timeouts = manifest.get("timeouts") if isinstance(manifest.get("timeouts"), dict) else {}
    startup_timeout = _timeout_value(
        manifest,
        timeouts,
        keys=("startup_timeout_sec",),
        nested_keys=("startup_seconds", "startup_timeout_sec"),
        nested_ms_keys=("initialize_ms", "startup_ms"),
        default=10,
    )
    tool_timeout = _timeout_value(
        manifest,
        timeouts,
        keys=("tool_timeout_sec",),
        nested_keys=("tool_seconds", "tool_timeout_sec"),
        nested_ms_keys=("tool_call_ms", "tool_ms"),
        default=60,
    )
    self_test_timeout = _timeout_value(
        manifest,
        timeouts,
        keys=("self_test_timeout_sec", "test_timeout_sec"),
        nested_keys=("self_test_seconds", "self_test_timeout_sec"),
        nested_ms_keys=("self_test_ms", "test_ms"),
        default=20,
    )
    cwd = manifest.get("cwd")
    if cwd is None:
        cwd_path = workspace
    else:
        cwd_path = Path(cwd)
        if not cwd_path.is_absolute():
            cwd_path = (workspace / cwd_path).resolve()
        elif cwd_path != workspace and _manifest_points_to_repo_local_server(workspace, manifest):
            cwd_path = workspace
    return {
        "name": name,
        "command": command,
        "args": args,
        "cwd": str(cwd_path),
        "startup_timeout_sec": int(startup_timeout),
        "tool_timeout_sec": int(tool_timeout),
        "self_test_timeout_sec": int(self_test_timeout),
        "self_test": _normalize_mcp_self_test(workspace, manifest, command, args, cwd_path),
        "enabled_tools": [str(tool) for tool in manifest.get("enabled_tools", [])],
    }


def _timeout_value(
    manifest: dict[str, Any],
    timeouts: dict[str, Any],
    *,
    keys: tuple[str, ...],
    nested_keys: tuple[str, ...],
    nested_ms_keys: tuple[str, ...],
    default: int,
) -> int:
    for key in keys:
        if manifest.get(key) is not None:
            return int(manifest[key])
    for key in nested_keys:
        if timeouts.get(key) is not None:
            return int(timeouts[key])
    for key in nested_ms_keys:
        if timeouts.get(key) is not None:
            return max(1, int(float(timeouts[key]) / 1000))
    return default


def _normalize_mcp_self_test(
    workspace: Path,
    manifest: dict[str, Any],
    command: str,
    args: list[str],
    server_cwd: Path,
) -> dict[str, Any] | None:
    if "self_test" not in manifest:
        return None
    raw = manifest.get("self_test")
    if raw is True:
        return {"command": [command, *args, "--self-test"], "cwd": str(server_cwd)}
    if not raw:
        return None
    if isinstance(raw, list):
        return {"command": [str(part) for part in raw], "cwd": str(server_cwd)}
    if isinstance(raw, str):
        return {"command": shlex.split(raw), "cwd": str(server_cwd)}
    if isinstance(raw, dict):
        raw_command = raw.get("command")
        raw_args = raw.get("args", [])
        if isinstance(raw_command, list):
            command_parts = [str(part) for part in raw_command]
        elif isinstance(raw_command, str):
            if raw_args:
                command_parts = [raw_command, *[str(part) for part in raw_args]]
            else:
                command_parts = shlex.split(raw_command)
        else:
            raise ValueError("self_test.command must be a string or list")
        spec: dict[str, Any] = {"command": command_parts, "cwd": str(server_cwd)}
        if raw.get("cwd") is not None:
            cwd_path = Path(str(raw["cwd"]))
            if not cwd_path.is_absolute():
                cwd_path = (workspace / cwd_path).resolve()
            elif cwd_path != workspace and _command_points_to_repo_local_path(workspace, command_parts):
                cwd_path = workspace
            spec["cwd"] = str(cwd_path)
        if raw.get("timeout_seconds") is not None:
            spec["timeout_sec"] = int(raw["timeout_seconds"])
        elif raw.get("timeout_sec") is not None:
            spec["timeout_sec"] = int(raw["timeout_sec"])
        elif raw.get("timeout_ms") is not None:
            spec["timeout_sec"] = max(1, int(float(raw["timeout_ms"]) / 1000))
        return spec
    raise ValueError("self_test must be true, false, a command string/list, or an object with command/args")


def _manifest_points_to_repo_local_server(workspace: Path, manifest: dict[str, Any]) -> bool:
    for arg in manifest.get("args", []):
        arg_path = Path(str(arg))
        if not arg_path.is_absolute() and (workspace / arg_path).exists():
            return True
    return False


def _command_points_to_repo_local_path(workspace: Path, command: list[str]) -> bool:
    for part in command[1:]:
        path = Path(part)
        if not path.is_absolute() and (workspace / path).exists():
            return True
    return False


def _smoke_mcp_server(server: dict[str, Any], timeout_seconds: float = 3.0) -> dict[str, Any]:
    command = [server["command"], *server.get("args", [])]
    try:
        process = subprocess.Popen(
            command,
            cwd=server["cwd"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        return {"status": "failed", "message": f"could not start server: {exc}"}
    try:
        _write_mcp_message(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "harnessgym", "version": "0.1"},
                },
            },
        )
        initialize = _read_mcp_message(process, timeout_seconds)
        if initialize is None:
            return {"status": "failed", "message": "no initialize response"}
        _write_mcp_message(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        _write_mcp_message(process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools = _read_mcp_message(process, timeout_seconds)
        if tools is None:
            return {"status": "failed", "message": "no tools/list response"}
        tool_names = [
            str(tool.get("name"))
            for tool in tools.get("result", {}).get("tools", [])
            if isinstance(tool, dict) and tool.get("name")
        ]
        return {"status": "passed", "tools": tool_names}
    except Exception as exc:
        return {"status": "failed", "message": str(exc)}
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def _run_mcp_self_test(server: dict[str, Any]) -> dict[str, Any]:
    spec = server.get("self_test")
    if not spec:
        return {"status": "not_configured", "message": "manifest has no self_test command"}
    command = spec.get("command") if isinstance(spec, dict) else spec
    cwd = spec.get("cwd", server["cwd"]) if isinstance(spec, dict) else server["cwd"]
    timeout_seconds = (
        float(spec.get("timeout_sec", server.get("self_test_timeout_sec", 20)))
        if isinstance(spec, dict)
        else float(server.get("self_test_timeout_sec", 20))
    )
    if not isinstance(command, list) or not command:
        return {"status": "failed", "message": "self_test command is empty or invalid", "command": command}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "message": f"self-test timed out after {timeout_seconds:g}s",
            "command": command,
            "stdout": _truncate_text(exc.stdout),
            "stderr": _truncate_text(exc.stderr),
        }
    except Exception as exc:
        return {"status": "failed", "message": f"could not run self-test: {exc}", "command": command}
    status = "passed" if completed.returncode == 0 else "failed"
    return {
        "status": status,
        "message": "self-test passed" if status == "passed" else f"self-test exited {completed.returncode}",
        "command": command,
        "return_code": completed.returncode,
        "stdout": _truncate_text(completed.stdout),
        "stderr": _truncate_text(completed.stderr),
    }


def _truncate_text(value: str | bytes | None, limit: int = 8000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def _write_mcp_message(process: subprocess.Popen, payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("MCP server stdin unavailable")
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    process.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
    process.stdin.flush()


def _read_mcp_message(process: subprocess.Popen, timeout_seconds: float) -> dict[str, Any] | None:
    if process.stdout is None:
        raise RuntimeError("MCP server stdout unavailable")
    deadline = time.monotonic() + timeout_seconds
    header_bytes = b""
    while b"\r\n\r\n" not in header_bytes and b"\n\n" not in header_bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([process.stdout], [], [], remaining)
        if not ready:
            return None
        chunk = os.read(process.stdout.fileno(), 1)
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
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        ready, _, _ = select.select([process.stdout], [], [], remaining)
        if not ready:
            return None
        body += os.read(process.stdout.fileno(), length - len(body))
    return json.loads(body[:length].decode("utf-8"))


def _safe_server_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return safe or "harnessgym_mcp"


def _write_project_codex_config(workspace: Path, servers: list[dict[str, Any]]) -> None:
    codex_dir = workspace / ".codex"
    config_path = codex_dir / "config.toml"
    if not servers:
        if config_path.exists():
            _replace_marked_block(config_path, "")
        return
    codex_dir.mkdir(parents=True, exist_ok=True)
    block_lines = [BEGIN_MARKER, "# Generated by HarnessGym. Edit .harnessgym/mcp manifests instead."]
    for server in servers:
        codex_config = _codex_server_config(server)
        block_lines.extend(
            [
                "",
                f"[mcp_servers.{_toml_key(server['name'])}]",
                f"command = {_toml_string(codex_config['command'])}",
                f"args = {_toml_array(codex_config['args'])}",
                f"cwd = {_toml_string(codex_config['cwd'])}",
                f"startup_timeout_sec = {server['startup_timeout_sec']}",
                f"tool_timeout_sec = {codex_config['tool_timeout_sec']}",
                "enabled = true",
            ]
        )
        if server["enabled_tools"]:
            block_lines.append(f"enabled_tools = {_toml_array(server['enabled_tools'])}")
    block_lines.append(END_MARKER)
    _replace_marked_block(config_path, "\n".join(block_lines) + "\n")


def _write_mcp_call_helper(workspace: Path) -> dict[str, Any]:
    helper_path = workspace / ".harnessgym" / "runtime" / "mcp_call.py"
    source_root = Path(__file__).resolve().parents[1]
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    helper_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "",
                "from pathlib import Path",
                "import runpy",
                "import sys",
                "",
                f"_SOURCE_ROOT = Path({str(source_root)!r})",
                "if _SOURCE_ROOT.exists():",
                "    sys.path.insert(0, str(_SOURCE_ROOT))",
                "",
                "runpy.run_module('harnessgym.mcp_client', run_name='__main__')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    helper_path.chmod(0o755)
    return {
        "path": str(helper_path),
        "relative_path": ".harnessgym/runtime/mcp_call.py",
        "usage": "python3 .harnessgym/runtime/mcp_call.py --server <name> --tool <tool> --arguments '<json-object>'",
    }


def _codex_server_config(server: dict[str, Any]) -> dict[str, Any]:
    tool_timeout = int(server.get("tool_timeout_sec", 60))
    return {
        "command": sys.executable,
        "args": [
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
        ],
        "cwd": str(server["cwd"]),
        "tool_timeout_sec": tool_timeout + 5,
    }


def _replace_marked_block(config_path: Path, block: str) -> None:
    current = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    pattern = re.compile(
        rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}\n?",
        re.DOTALL,
    )
    stripped = pattern.sub("", current).rstrip()
    pieces = [piece for piece in [stripped, block.rstrip()] if piece]
    config_path.write_text("\n\n".join(pieces) + ("\n" if pieces else ""), encoding="utf-8")


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _toml_key(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return _toml_string(value)
