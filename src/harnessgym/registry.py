from __future__ import annotations

import json
from pathlib import Path

from .models import Artifact, Registry, utc_now


def registry_path(workspace: Path) -> Path:
    return workspace / ".harnessgym" / "registry.json"


def load_registry(workspace: Path) -> Registry:
    path = registry_path(workspace)
    if not path.exists():
        return Registry()
    return Registry.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_registry(workspace: Path, registry: Registry) -> None:
    path = registry_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.updated_at = utc_now()
    path.write_text(json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_or_update_artifact(registry: Registry, artifact: Artifact) -> None:
    existing = registry.get_artifact(artifact.id)
    if existing is None:
        registry.artifacts.append(artifact)
        return
    existing.kind = artifact.kind
    existing.path = artifact.path
    existing.description = artifact.description or existing.description
    existing.iteration = artifact.iteration if artifact.iteration is not None else existing.iteration
    existing.metadata.update(artifact.metadata)


def artifact_is_quarantined(artifact: Artifact) -> bool:
    quarantine = artifact.metadata.get("quarantine")
    if isinstance(quarantine, dict) and quarantine.get("status") == "quarantined":
        return True
    return bool(artifact.metadata.get("quarantined"))


def active_artifacts(registry: Registry) -> list[Artifact]:
    return [artifact for artifact in registry.artifacts if not artifact_is_quarantined(artifact)]


def active_registry(registry: Registry) -> Registry:
    return Registry(
        version=registry.version,
        artifacts=list(active_artifacts(registry)),
        updated_at=registry.updated_at,
        metadata=dict(registry.metadata),
    )


def quarantine_artifacts(
    registry: Registry,
    *,
    paths: list[str],
    reason: str,
    report_path: str | None = None,
    iteration: int | None = None,
) -> list[str]:
    path_set = set(paths)
    quarantined: list[str] = []
    for artifact in registry.artifacts:
        if artifact.path not in path_set:
            continue
        artifact.metadata["quarantined"] = True
        artifact.metadata["quarantine"] = {
            "status": "quarantined",
            "reason": reason,
            "report_path": report_path,
            "iteration": iteration,
            "updated_at": utc_now(),
        }
        quarantined.append(artifact.path)
    return sorted(dict.fromkeys(quarantined))


def mark_artifacts_qualified(
    registry: Registry,
    *,
    paths: list[str],
    report_path: str | None = None,
    iteration: int | None = None,
) -> list[str]:
    path_set = set(paths)
    qualified: list[str] = []
    for artifact in registry.artifacts:
        if artifact.path not in path_set:
            continue
        artifact.metadata.pop("quarantined", None)
        artifact.metadata.pop("quarantine", None)
        artifact.metadata["qualification"] = {
            "status": "passed",
            "report_path": report_path,
            "iteration": iteration,
            "updated_at": utc_now(),
        }
        qualified.append(artifact.path)
    return sorted(dict.fromkeys(qualified))
