from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import Artifact, Registry, utc_now
from .registry import add_or_update_artifact, save_registry


ARTIFACT_DIRS = {
    "skill": "skills",
    "mcp": "mcp",
    "tool": "tools",
    "verifier": "verifiers",
    "fixture": "fixtures",
    "test": "tests",
    "docs": "docs",
    "script": "scripts",
}


def ensure_harness_dirs(workspace: Path) -> Path:
    harness_dir = workspace / ".harnessgym"
    for dirname in ["runs", *ARTIFACT_DIRS.values()]:
        (harness_dir / dirname).mkdir(parents=True, exist_ok=True)
    return harness_dir


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def run_dir(workspace: Path, run_id: str) -> Path:
    return workspace / ".harnessgym" / "runs" / run_id


def iteration_dir(workspace: Path, run_id: str, iteration: int) -> Path:
    return run_dir(workspace, run_id) / "iterations" / str(iteration)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def deep_merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def create_initial_result(
    *,
    path: Path,
    run_id: str,
    iteration: int,
    task_path: Path | None,
    registry: Registry,
) -> None:
    data = {
        "run_id": run_id,
        "iteration": iteration,
        "status": "running",
        "verified": False,
        "task_path": str(task_path) if task_path else None,
        "registry_artifact_count": len(registry.artifacts),
        "phases": {},
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    write_json(path, data)


def update_result(path: Path, updates: dict) -> dict:
    current = read_json(path)
    merged = deep_merge(current, updates)
    merged["updated_at"] = utc_now()
    write_json(path, merged)
    return merged


def sync_registry_from_files(workspace: Path, registry: Registry, iteration: int | None) -> Registry:
    harness_dir = ensure_harness_dirs(workspace)
    for kind, dirname in ARTIFACT_DIRS.items():
        root = harness_dir / dirname
        for artifact_path in sorted(root.rglob("*")):
            if not artifact_path.is_file():
                continue
            if "__pycache__" in artifact_path.parts or artifact_path.suffix in {".pyc", ".pyo"}:
                continue
            if kind == "skill" and artifact_path.name != "SKILL.md":
                continue
            if kind == "mcp" and artifact_path.name not in {"mcp.json", "server.json", "harnessgym-mcp.json"}:
                continue
            rel_path = artifact_path.relative_to(workspace).as_posix()
            artifact_id = f"{kind}:{rel_path}"
            if registry.get_artifact(artifact_id) is not None:
                continue
            artifact = Artifact(
                id=artifact_id,
                kind=kind,
                path=rel_path,
                description=f"Discovered {kind} artifact at {rel_path}",
                iteration=iteration,
                metadata={"source": "filesystem-sync"},
            )
            add_or_update_artifact(registry, artifact)
    save_registry(workspace, registry)
    return registry
