from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .artifacts import EXCLUDED_TEMPLATE_NAMES as EXCLUDED_CHECKPOINT_NAMES
from .artifacts import EXCLUDED_TEMPLATE_SUFFIXES as EXCLUDED_CHECKPOINT_SUFFIXES
from .artifacts import write_json
from .models import utc_now


class BestCheckpointManager:
    """Capture and restore the best independently scored task workspace."""

    def __init__(self, workspace: Path, run_dir: Path) -> None:
        self.workspace = workspace
        self.root = run_dir / "checkpoints"
        self.best_dir = self.root / "best"
        self.manifest_path = self.root / "best_manifest.json"

    def capture(self, *, iteration: int, score: float, reason: str) -> dict[str, Any]:
        if self.best_dir.exists():
            shutil.rmtree(self.best_dir)
        self.best_dir.mkdir(parents=True, exist_ok=True)

        files: list[str] = []
        for source in self._iter_workspace_paths():
            rel_path = source.relative_to(self.workspace)
            target = self.best_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(Path(source.readlink()))
            elif source.is_file():
                shutil.copy2(source, target)
                files.append(rel_path.as_posix())
            elif source.is_dir():
                target.mkdir(exist_ok=True)

        manifest = {
            "created_at": utc_now(),
            "iteration": iteration,
            "score": score,
            "reason": reason,
            "checkpoint_path": str(self.best_dir),
            "manifest_path": str(self.manifest_path),
            "file_count": len(files),
            "files": sorted(files),
        }
        write_json(self.manifest_path, manifest)
        return manifest

    def restore(self) -> dict[str, Any]:
        if not self.best_dir.exists():
            raise RuntimeError("best checkpoint restore requested but no best checkpoint exists")
        for path in sorted(self.workspace.iterdir(), key=lambda candidate: candidate.name):
            if _is_excluded(path.relative_to(self.workspace)):
                continue
            _remove(path)

        restored_files: list[str] = []
        for source in self._iter_snapshot_paths():
            rel_path = source.relative_to(self.best_dir)
            target = self.workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(Path(source.readlink()))
            elif source.is_file():
                shutil.copy2(source, target)
                restored_files.append(rel_path.as_posix())
            elif source.is_dir():
                target.mkdir(exist_ok=True)

        report: dict[str, Any] = {
            "restored": True,
            "restored_at": utc_now(),
            "checkpoint_path": str(self.best_dir),
            "manifest_path": str(self.manifest_path),
            "restored_file_count": len(restored_files),
            "restored_files": sorted(restored_files),
        }
        return report

    def has_checkpoint(self) -> bool:
        return self.best_dir.exists()

    def _iter_workspace_paths(self) -> list[Path]:
        paths: list[Path] = []
        for path in self.workspace.rglob("*"):
            try:
                rel_path = path.relative_to(self.workspace)
            except ValueError:
                continue
            if _is_excluded(rel_path):
                continue
            paths.append(path)
        return sorted(paths, key=lambda candidate: (len(candidate.parts), str(candidate)))

    def _iter_snapshot_paths(self) -> list[Path]:
        return sorted(self.best_dir.rglob("*"), key=lambda candidate: (len(candidate.parts), str(candidate)))


def _is_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_CHECKPOINT_NAMES:
        return True
    return rel_path.suffix in EXCLUDED_CHECKPOINT_SUFFIXES


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
