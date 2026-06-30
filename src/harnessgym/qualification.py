from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .activation import activate_generated_harness
from .artifacts import ARTIFACT_DIRS, write_json
from .models import Registry, utc_now
from .registry import load_registry


EXCLUDED_TEMPLATE_NAMES = {
    ".agents",
    ".codex",
    ".git",
    ".harnessgym",
    ".harnessgym_build",
    ".pytest_cache",
    "__pycache__",
}
EXCLUDED_TEMPLATE_SUFFIXES = {".o", ".pyc", ".pyo", ".so", ".dylib"}


def capture_clean_workspace_template(workspace: Path, destination: Path) -> dict[str, Any]:
    """Snapshot non-harness task files for later fresh artifact qualification."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for source in _iter_copyable_paths(workspace):
        rel_path = source.relative_to(workspace)
        target = destination / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(Path(source.readlink()))
        elif source.is_file():
            shutil.copy2(source, target)
            copied_files.append(rel_path.as_posix())
        elif source.is_dir():
            target.mkdir(exist_ok=True)
    return {
        "created_at": utc_now(),
        "workspace": str(workspace),
        "template_path": str(destination),
        "file_count": len(copied_files),
        "files": sorted(copied_files),
    }


def qualify_generated_harness(
    *,
    source_workspace: Path,
    template_dir: Path,
    qualification_dir: Path,
    iteration: int,
    registry: Registry,
) -> dict[str, Any]:
    """Activate generated harness artifacts in a clean copied workspace."""
    qualification_dir.mkdir(parents=True, exist_ok=True)
    fresh_workspace = qualification_dir / "workspace"
    if fresh_workspace.exists():
        shutil.rmtree(fresh_workspace)
    shutil.copytree(template_dir, fresh_workspace, symlinks=True)
    copied_artifacts = copy_reusable_harness(source_workspace, fresh_workspace)
    fresh_registry = _candidate_registry_for_qualification(load_registry(fresh_workspace))
    activation = activate_generated_harness(fresh_workspace, fresh_registry)
    quality_gate = activation.get("quality_gate") if isinstance(activation, dict) else {}
    status = "passed" if isinstance(quality_gate, dict) and quality_gate.get("status") == "passed" else "failed"
    failed_artifacts = _failed_artifact_paths(activation, registry)
    report = {
        "created_at": utc_now(),
        "status": status,
        "iteration": iteration,
        "source_workspace": str(source_workspace),
        "fresh_workspace": str(fresh_workspace),
        "copied_artifacts": copied_artifacts,
        "quality_gate": quality_gate,
        "activation": activation,
        "failed_artifacts": failed_artifacts,
        "registry_artifact_count": len(registry.artifacts),
    }
    write_json(qualification_dir / "qualification.json", report)
    return report


def _candidate_registry_for_qualification(registry: Registry) -> Registry:
    data = registry.to_dict()
    for artifact in data.get("artifacts", []):
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("quarantined", None)
            metadata.pop("quarantine", None)
    return Registry.from_dict(data)


def copy_reusable_harness(source_workspace: Path, destination_workspace: Path) -> list[str]:
    source_harness = source_workspace / ".harnessgym"
    destination_harness = destination_workspace / ".harnessgym"
    destination_harness.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for dirname in ARTIFACT_DIRS.values():
        source = source_harness / dirname
        if not source.exists():
            continue
        destination = destination_harness / dirname
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, ignore=_artifact_ignore)
        copied.extend(
            path.relative_to(destination_workspace).as_posix()
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        )
    registry_path = source_harness / "registry.json"
    if registry_path.exists():
        target = destination_harness / "registry.json"
        shutil.copy2(registry_path, target)
        copied.append(target.relative_to(destination_workspace).as_posix())
    return sorted(copied)


def _failed_artifact_paths(activation: dict[str, Any], registry: Registry) -> list[str]:
    paths: list[str] = []
    for server in activation.get("mcp_servers", []) if isinstance(activation, dict) else []:
        if not isinstance(server, dict):
            continue
        if server.get("active") is False and server.get("artifact_path"):
            paths.append(str(server["artifact_path"]))
    return sorted(dict.fromkeys(paths))


def _iter_copyable_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            continue
        if _is_excluded(rel_path):
            continue
        paths.append(path)
    return sorted(paths, key=lambda candidate: (len(candidate.parts), str(candidate)))


def _is_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_TEMPLATE_NAMES:
        return True
    return rel_path.suffix in EXCLUDED_TEMPLATE_SUFFIXES


def _artifact_ignore(directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "__pycache__" or (Path(directory) / name).suffix in {".pyc", ".pyo"}
    }
