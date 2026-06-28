from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


def run_post_command(
    *,
    command: str,
    cwd: Path,
    log_dir: Path,
    timeout_seconds: int,
    score_key: str,
    prefix: str = "post",
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{prefix}.stdout.txt"
    stderr_path = log_dir / f"{prefix}.stderr.txt"
    started = time.monotonic()
    stdout = ""
    stderr = ""
    return_code: int | None = None
    timed_out = False
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _timeout_text(exc.stdout)
        stderr = _timeout_text(exc.stderr) + f"\nTimed out after {timeout_seconds} seconds.\n"

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    payload = json_object(stdout)
    status = "timeout" if timed_out else ("completed" if return_code == 0 else "failed")
    return {
        "command": command,
        "status": status,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "json": payload,
        "score": extract_score_from_payload(payload or {}, score_key),
        "score_key": score_key,
    }


def post_command_passed(result: dict[str, Any]) -> bool:
    if result.get("status") != "completed":
        return False
    payload = result.get("json")
    if not isinstance(payload, dict):
        return True
    payload_status = payload.get("status")
    if payload_status is None:
        return True
    return str(payload_status).lower() in {"passed", "pass", "ok", "success", "solved"}


def json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_score_from_payload(result_data: dict[str, Any], score_key: str) -> float | None:
    candidates: list[Any] = [
        result_data.get(score_key),
        result_data.get("score"),
    ]
    metrics = result_data.get("metrics")
    if isinstance(metrics, dict):
        candidates.extend([metrics.get(score_key), metrics.get("score")])
    objective = result_data.get("objective")
    if isinstance(objective, dict):
        candidates.extend([objective.get(score_key), objective.get("score")])
    verification = result_data.get("verification")
    if isinstance(verification, dict):
        candidates.extend([verification.get(score_key), verification.get("score")])
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)
        if isinstance(candidate, str):
            try:
                return float(candidate)
            except ValueError:
                continue
    return None


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
