from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Artifact:
    id: str
    kind: str
    path: str
    description: str = ""
    iteration: int | None = None
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        kind = str(data["kind"])
        path = str(data["path"])
        return cls(
            id=str(data.get("id") or f"{kind}:{path}"),
            kind=kind,
            path=path,
            description=str(data.get("description", "")),
            iteration=data.get("iteration"),
            created_at=str(data.get("created_at") or utc_now()),
            metadata={key: value for key, value in data.items() if key not in {
                "id",
                "kind",
                "path",
                "description",
                "iteration",
                "created_at",
                "metadata",
            }} | dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Registry:
    version: int = 1
    artifacts: list[Artifact] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Registry":
        return cls(
            version=int(data.get("version", 1)),
            artifacts=[Artifact.from_dict(item) for item in data.get("artifacts", [])],
            updated_at=str(data.get("updated_at") or utc_now()),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": self.metadata,
        }

    def get_artifact(self, artifact_id: str) -> Artifact | None:
        for artifact in self.artifacts:
            if artifact.id == artifact_id:
                return artifact
        return None


@dataclass
class RunnerResult:
    phase: str
    status: str
    session_id: str | None = None
    return_code: int | None = None
    timed_out: bool = False
    duration_seconds: float = 0.0
    stdout_path: str | None = None
    stderr_path: str | None = None
    transcript_path: str | None = None
    prompt_path: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IterationResult:
    iteration: int
    status: str
    result_path: str
    attempt: RunnerResult | None = None
    reflection: RunnerResult | None = None
    build: RunnerResult | None = None
    solved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IterationContext:
    run_id: str
    iteration: int
    workspace: Path
    harness_dir: Path
    run_dir: Path
    iteration_dir: Path
    result_path: Path
    registry: Registry
    task_text: str
    task_path: Path | None = None
    artifact_context: str = ""
    attempt_timeout_seconds: int | None = None
