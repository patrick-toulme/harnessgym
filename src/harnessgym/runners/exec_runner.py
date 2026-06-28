from __future__ import annotations

import json
import os
import re
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


SESSION_PATTERNS = [
    re.compile(r"session(?:[_ -]?id)?\s*[:=]\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE),
    re.compile(r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", re.IGNORECASE),
]


def parse_session_id(text: str) -> str | None:
    for pattern in SESSION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


class ExecRunner(Runner):
    """Runner backend using `codex exec` and `codex exec resume <session_id>`."""

    def __init__(self, codex_bin: str = "codex") -> None:
        self.codex_bin = codex_bin

    def build_command(
        self,
        prompt: str,
        session_id: str | None = None,
        workspace: Path | None = None,
    ) -> list[str]:
        config_overrides = self._project_mcp_config_overrides(workspace)
        if session_id:
            return [self.codex_bin, "exec", "resume", *config_overrides, session_id, prompt]
        return [self.codex_bin, "exec", *config_overrides, prompt]

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
            message = f"Cannot run {phase}: no Codex session id is available to resume."
            stderr_path.write_text(message + "\n", encoding="utf-8")
            stdout_path.write_text("", encoding="utf-8")
            self._write_transcript(
                path=transcript_path,
                command=[self.codex_bin, "exec", "resume", "<missing-session-id>", prompt],
                prompt_path=prompt_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                status="failed",
                return_code=None,
                timed_out=False,
                duration=0.0,
                session_id=None,
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
                        stderr = (stderr or "") + "\nTimed out while collecting Codex output after kill.\n"
                stderr = (stderr or "") + f"\nTimed out after {timeout_seconds} seconds; terminated Codex process.\n"
        except FileNotFoundError as exc:
            return_code = 127
            stderr = str(exc)
            message = f"Codex executable not found: {self.codex_bin}"

        duration = time.monotonic() - started
        parsed_session = parse_session_id(f"{stdout}\n{stderr}") or session_id
        status = "timeout" if timed_out else ("completed" if return_code == 0 else "failed")

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
    ) -> None:
        display_command = command[:-1] + ["<prompt; see prompt file>"]
        path.write_text(
            "\n".join(
                [
                    f"command: {shlex.join(display_command)}",
                    f"prompt_path: {prompt_path}",
                    f"stdout_path: {stdout_path}",
                    f"stderr_path: {stderr_path}",
                    f"status: {status}",
                    f"return_code: {return_code}",
                    f"timed_out: {timed_out}",
                    f"duration_seconds: {duration:.3f}",
                    f"session_id: {session_id or ''}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

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

    def _project_mcp_config_overrides(self, workspace: Path | None) -> list[str]:
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
        overrides: list[str] = []
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
            name = str(server.get("name") or "")
            if not name:
                continue
            prefix = f"mcp_servers.{self._config_key(name)}"
            for key, value in self._server_config_values(server).items():
                overrides.extend(["-c", f"{prefix}.{key}={self._toml_value(value)}"])
        return overrides

    def _server_config_values(self, server: dict[str, Any]) -> dict[str, Any]:
        tool_timeout = int(server.get("tool_timeout_sec", 60))
        values: dict[str, Any] = {
            "command": sys.executable,
            "args": [
                str(Path(__file__).resolve().parents[1] / "mcp_telemetry_proxy.py"),
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
            "startup_timeout_sec": int(server.get("startup_timeout_sec", 10)),
            "tool_timeout_sec": tool_timeout + 5,
            "enabled": True,
        }
        enabled_tools = [str(tool) for tool in server.get("enabled_tools", [])]
        if enabled_tools:
            values["enabled_tools"] = enabled_tools
        return values

    def _config_key(self, value: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+", value):
            return value
        return json.dumps(value)

    def _toml_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(self._toml_value(item) for item in value) + "]"
        return json.dumps(str(value))
