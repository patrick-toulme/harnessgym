from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from harnessgym.config import RunConfig
from harnessgym.models import IterationContext, RunnerResult

from .base import Runner
from .exec_runner import parse_session_id


def parse_claude_result(text: str) -> dict[str, Any] | None:
    """Parse Claude Code --output-format json, with JSONL tolerance for future output."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "result":
            return payload
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("session_id"):
            return payload
    return None


def parse_claude_session_id(text: str) -> str | None:
    payload = parse_claude_result(text)
    if payload is not None and payload.get("session_id"):
        return str(payload["session_id"])
    return parse_session_id(text)


class ClaudeRunner(Runner):
    """Runner backend using Claude Code print mode and same-session resume."""

    def __init__(
        self,
        claude_bin: str = "claude",
        *,
        model: str | None = None,
        permission_mode: str = "bypassPermissions",
        max_budget_usd: float | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.claude_bin = claude_bin
        self.model = model
        self.permission_mode = permission_mode
        self.max_budget_usd = max_budget_usd
        self.extra_args = tuple(extra_args)

    def build_command(
        self,
        prompt: str,
        session_id: str | None = None,
        workspace: Path | None = None,
    ) -> list[str]:
        command = [self.claude_bin, "-p", "--output-format", "json"]
        if self.permission_mode:
            command.extend(["--permission-mode", self.permission_mode])
        if self.model:
            command.extend(["--model", self.model])
        if self.max_budget_usd is not None:
            command.extend(["--max-budget-usd", f"{self.max_budget_usd:g}"])
        command.extend(self.extra_args)

        mcp_config = self._write_project_mcp_config(workspace)
        if mcp_config is not None:
            command.extend(["--strict-mcp-config", "--mcp-config", str(mcp_config)])
            allowed_tools = self._allowed_mcp_tool_patterns(workspace)
            if allowed_tools:
                command.append(f"--allowedTools={','.join(allowed_tools)}")
        if session_id:
            command.extend(["--resume", session_id])
        command.append(prompt)
        return command

    def start_attempt(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
    ) -> RunnerResult:
        return self._run_phase(
            config=config,
            context=context,
            phase="attempt",
            prompt=prompt,
            timeout_seconds=context.attempt_timeout_seconds or config.attempt_timeout_seconds,
            session_id=None,
        )

    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        return self._run_phase(
            config=config,
            context=context,
            phase="reflection",
            prompt=prompt,
            timeout_seconds=config.reflection_timeout_seconds,
            session_id=session_id,
        )

    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        return self._run_phase(
            config=config,
            context=context,
            phase="build",
            prompt=prompt,
            timeout_seconds=config.build_timeout_seconds,
            session_id=session_id,
        )

    def _run_phase(
        self,
        *,
        config: RunConfig,
        context: IterationContext,
        phase: str,
        prompt: str,
        timeout_seconds: int,
        session_id: str | None,
    ) -> RunnerResult:
        prompt_path = context.iteration_dir / f"{phase}.prompt.txt"
        stdout_path = context.iteration_dir / f"{phase}.stdout.txt"
        stderr_path = context.iteration_dir / f"{phase}.stderr.txt"
        transcript_path = context.iteration_dir / f"{phase}.transcript.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        if phase in {"reflection", "build"} and not session_id:
            message = f"Cannot run {phase}: no Claude Code session id is available to resume."
            stderr_path.write_text(message + "\n", encoding="utf-8")
            stdout_path.write_text("", encoding="utf-8")
            self._write_transcript(
                path=transcript_path,
                command=[self.claude_bin, "-p", "--output-format", "json", "--resume", "<missing-session-id>", prompt],
                prompt_path=prompt_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                status="failed",
                return_code=None,
                timed_out=False,
                duration=0.0,
                session_id=None,
                claude_result=None,
            )
            return RunnerResult(
                phase=phase,
                status="failed",
                return_code=None,
                duration_seconds=0.0,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                transcript_path=str(transcript_path),
                prompt_path=str(prompt_path),
                message=message,
            )

        command = self.build_command(prompt, session_id=session_id, workspace=config.workspace)
        started = time.monotonic()
        stdout = ""
        stderr = ""
        return_code: int | None = None
        timed_out = False
        message = ""

        try:
            process = subprocess.Popen(
                command,
                cwd=config.workspace,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_process(process, signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self._terminate_process(process, signal.SIGKILL)
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        stdout = stdout or ""
                        stderr = (stderr or "") + "\nTimed out while collecting Claude Code output after kill.\n"
                stderr = (stderr or "") + f"\nTimed out after {timeout_seconds} seconds; terminated Claude Code process.\n"
        except FileNotFoundError as exc:
            return_code = 127
            stderr = str(exc)
            message = f"Claude Code executable not found: {self.claude_bin}"

        duration = time.monotonic() - started
        claude_result = parse_claude_result(stdout)
        parsed_session = parse_claude_session_id(f"{stdout}\n{stderr}") or session_id
        status = self._phase_status(timed_out, return_code, claude_result)
        if not message and isinstance(claude_result, dict):
            result_text = claude_result.get("result")
            if isinstance(result_text, str):
                message = result_text[:1000]

        stdout_path.write_text(stdout or "", encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")
        self._write_transcript(
            path=transcript_path,
            command=command,
            prompt_path=prompt_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            status=status,
            return_code=return_code,
            timed_out=timed_out,
            duration=duration,
            session_id=parsed_session,
            claude_result=claude_result,
        )

        return RunnerResult(
            phase=phase,
            status=status,
            session_id=parsed_session,
            return_code=return_code,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            transcript_path=str(transcript_path),
            prompt_path=str(prompt_path),
            message=message,
        )

    def _phase_status(
        self,
        timed_out: bool,
        return_code: int | None,
        claude_result: dict[str, Any] | None,
    ) -> str:
        if timed_out:
            return "timeout"
        if return_code != 0:
            return "failed"
        if claude_result is not None and claude_result.get("is_error") is True:
            return "failed"
        subtype = claude_result.get("subtype") if isinstance(claude_result, dict) else None
        if subtype in {"error", "failed"}:
            return "failed"
        return "completed"

    def _write_transcript(
        self,
        *,
        path: Path,
        command: list[str],
        prompt_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        status: str,
        return_code: int | None,
        timed_out: bool,
        duration: float,
        session_id: str | None,
        claude_result: dict[str, Any] | None,
    ) -> None:
        display_command = command[:-1] + ["<prompt; see prompt file>"]
        lines = [
            f"command: {shlex.join(display_command)}",
            f"prompt_path: {prompt_path}",
            f"stdout_path: {stdout_path}",
            f"stderr_path: {stderr_path}",
            f"status: {status}",
            f"return_code: {return_code}",
            f"timed_out: {timed_out}",
            f"duration_seconds: {duration:.3f}",
            f"session_id: {session_id or ''}",
        ]
        if isinstance(claude_result, dict):
            for key in (
                "type",
                "subtype",
                "is_error",
                "api_error_status",
                "stop_reason",
                "terminal_reason",
                "total_cost_usd",
            ):
                if key in claude_result:
                    lines.append(f"claude_{key}: {claude_result[key]}")
            result_text = claude_result.get("result")
            if isinstance(result_text, str) and result_text.strip():
                lines.extend(["", "claude_result:", result_text[:8000]])
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _terminate_process(self, process: subprocess.Popen[str], sig: signal.Signals) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except Exception:
            if sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()

    def _write_project_mcp_config(self, workspace: Path | None) -> Path | None:
        servers = self._active_project_mcp_servers(workspace)
        if not servers or workspace is None:
            return None
        config_path = workspace / ".harnessgym" / "claude_mcp_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mcpServers": {
                server["name"]: {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [
                        str(Path(__file__).resolve().parents[1] / "claude_mcp_bridge.py"),
                        "--server-name",
                        str(server["name"]),
                        "--response-timeout",
                        str(server.get("tool_timeout_sec", 30)),
                        "--cwd",
                        str(server.get("cwd") or workspace),
                        "--",
                        self._claude_command(server, workspace),
                        *[self._claude_arg(arg, workspace) for arg in server.get("args", [])],
                    ],
                    "env": {
                        "HARNESSGYM_WORKSPACE": str(workspace),
                        "CLAUDE_PROJECT_DIR": str(workspace),
                    },
                }
                for server in servers
            }
        }
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return config_path

    def _claude_command(self, server: dict[str, Any], workspace: Path) -> str:
        command = str(server["command"])
        command_path = Path(command)
        if not command_path.is_absolute() and (workspace / command_path).exists():
            return str((workspace / command_path).resolve())
        return command

    def _claude_arg(self, arg: Any, workspace: Path) -> str:
        value = str(arg)
        path = Path(value)
        if not path.is_absolute() and (workspace / path).exists():
            return str((workspace / path).resolve())
        return value

    def _allowed_mcp_tool_patterns(self, workspace: Path | None) -> list[str]:
        # Claude Code does not support MCP permission wildcards. The server-level
        # token approves all tools from that MCP server.
        return [f"mcp__{server['name']}" for server in self._active_project_mcp_servers(workspace)]

    def _active_project_mcp_servers(self, workspace: Path | None) -> list[dict[str, Any]]:
        if workspace is None:
            return []
        activation_path = workspace / ".harnessgym" / "activation.json"
        if not activation_path.exists():
            return []
        try:
            activation = json.loads(activation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        servers = activation.get("mcp_servers", []) if isinstance(activation, dict) else []
        active_servers: list[dict[str, Any]] = []
        for server in servers:
            if not isinstance(server, dict):
                continue
            if server.get("active") is False:
                continue
            smoke = server.get("smoke") if isinstance(server.get("smoke"), dict) else {}
            self_test = server.get("self_test") if isinstance(server.get("self_test"), dict) else {}
            if smoke.get("status") not in {None, "passed"}:
                continue
            if self_test.get("status") not in {None, "passed"}:
                continue
            if not server.get("name") or not server.get("command"):
                continue
            active_servers.append(server)
        return active_servers
