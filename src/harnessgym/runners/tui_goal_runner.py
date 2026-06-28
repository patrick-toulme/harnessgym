from __future__ import annotations

import os
import pty
import select
import subprocess
import time

from harnessgym.artifacts import read_json
from harnessgym.config import RunConfig
from harnessgym.models import IterationContext, RunnerResult

from .base import Runner


class TuiGoalRunner(Runner):
    """Experimental PTY backend that sends a real `/goal` for the attempt phase."""

    def __init__(self, codex_bin: str = "codex") -> None:
        self.codex_bin = codex_bin
        self.process: subprocess.Popen | None = None
        self.master_fd: int | None = None

    def start_attempt(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
    ) -> RunnerResult:
        self._start_process(config)
        prompt_path = context.iteration_dir / "attempt.prompt.txt"
        command_text = f"/goal Read and follow the HarnessGym attempt prompt at {prompt_path}."
        timeout_seconds = context.attempt_timeout_seconds or config.attempt_timeout_seconds
        return self._send_phase(config, context, "attempt", prompt, command_text, timeout_seconds)

    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        prompt_path = context.iteration_dir / "reflection.prompt.txt"
        command_text = f"Read and follow the HarnessGym reflection prompt at {prompt_path}."
        return self._send_phase(config, context, "reflection", prompt, command_text, config.reflection_timeout_seconds)

    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        prompt_path = context.iteration_dir / "build.prompt.txt"
        command_text = f"Read and follow the HarnessGym build prompt at {prompt_path}."
        return self._send_phase(config, context, "build", prompt, command_text, config.build_timeout_seconds)

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
        self.process = None
        self.master_fd = None

    def _start_process(self, config: RunConfig) -> None:
        if self.process and self.process.poll() is None:
            return
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        self.process = subprocess.Popen(
            [self.codex_bin],
            cwd=config.workspace,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
            close_fds=True,
        )
        os.close(slave_fd)

    def _send_phase(
        self,
        config: RunConfig,
        context: IterationContext,
        phase: str,
        prompt: str,
        command_text: str,
        timeout_seconds: int,
    ) -> RunnerResult:
        if self.master_fd is None or self.process is None:
            raise RuntimeError("TUI process is not running")

        prompt_path = context.iteration_dir / f"{phase}.prompt.txt"
        stdout_path = context.iteration_dir / f"{phase}.stdout.txt"
        stderr_path = context.iteration_dir / f"{phase}.stderr.txt"
        transcript_path = context.iteration_dir / f"{phase}.transcript.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        started = time.monotonic()
        output_chunks: list[str] = []
        os.write(self.master_fd, (command_text + "\n").encode("utf-8"))
        timed_out = False

        deadline = started + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            ready, _, _ = select.select([self.master_fd], [], [], 0.25)
            if ready:
                try:
                    output_chunks.append(os.read(self.master_fd, 8192).decode("utf-8", errors="replace"))
                except OSError:
                    break
            if self._phase_complete(context, phase):
                break
        else:
            timed_out = True
            os.write(self.master_fd, b"\x03")

        stdout = "".join(output_chunks)
        status = "timeout" if timed_out else "completed"
        duration = time.monotonic() - started
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        transcript_path.write_text(
            "\n".join(
                [
                    "runner: tui-goal",
                    f"command_text: {command_text}",
                    f"status: {status}",
                    f"timed_out: {timed_out}",
                    f"duration_seconds: {duration:.3f}",
                    "note: experimental PTY automation; completion is inferred from result.json.",
                    "",
                    stdout,
                ]
            ),
            encoding="utf-8",
        )
        return RunnerResult(
            phase=phase,
            status=status,
            timed_out=timed_out,
            duration_seconds=duration,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            transcript_path=str(transcript_path),
            prompt_path=str(prompt_path),
            message="Experimental TUI backend; use exec for the MVP path.",
        )

    def _phase_complete(self, context: IterationContext, phase: str) -> bool:
        data = read_json(context.result_path)
        if phase == "attempt":
            return data.get("status") not in (None, "running")
        if phase == "reflection":
            reflection = data.get("reflection")
            return isinstance(reflection, dict) and bool(reflection.get("selected_improvement"))
        if phase == "build":
            return bool(data.get("tooling")) or data.get("status") == "tooling_built"
        return False
