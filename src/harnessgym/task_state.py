from __future__ import annotations

import shutil
from pathlib import Path


TASK_STATE_CHOICES = ("continue", "reset")


class TaskStateManager:
    """Manage whether task edits compound across iterations."""

    def __init__(self, workspace: Path, mode: str) -> None:
        if mode not in TASK_STATE_CHOICES:
            choices = ", ".join(TASK_STATE_CHOICES)
            raise ValueError(f"task_state must be one of: {choices}")
        self.workspace = workspace
        self.mode = mode
        self.snapshot_dir = workspace / ".harnessgym" / "task_state" / "initial"

    def capture_initial(self) -> None:
        if self.mode != "reset":
            return
        if self.snapshot_dir.exists():
            shutil.rmtree(self.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        for path in self._iter_task_paths(self.workspace):
            rel_path = path.relative_to(self.workspace)
            target = self.snapshot_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                target.symlink_to(Path(path.readlink()))
            elif path.is_file():
                shutil.copy2(path, target)
            elif path.is_dir():
                target.mkdir(exist_ok=True)

    def prepare_iteration(self, iteration: int) -> None:
        if self.mode != "reset" or iteration == 1:
            return
        self._restore_snapshot()

    def _restore_snapshot(self) -> None:
        if not self.snapshot_dir.exists():
            raise RuntimeError("task-state reset requested but no initial snapshot exists")
        for path in sorted(self.workspace.iterdir(), key=lambda candidate: candidate.name):
            if path.name == ".harnessgym":
                continue
            _remove(path)
        for source in self._iter_snapshot_paths(self.snapshot_dir):
            rel_path = source.relative_to(self.snapshot_dir)
            target = self.workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink():
                target.symlink_to(Path(source.readlink()))
            elif source.is_file():
                shutil.copy2(source, target)
            elif source.is_dir():
                target.mkdir(exist_ok=True)

    def _iter_task_paths(self, root: Path) -> list[Path]:
        paths: list[Path] = []
        for path in root.rglob("*"):
            try:
                rel_parts = path.relative_to(root).parts
            except ValueError:
                continue
            if not rel_parts or rel_parts[0] == ".harnessgym":
                continue
            paths.append(path)
        return sorted(paths, key=lambda candidate: (len(candidate.parts), str(candidate)))

    def _iter_snapshot_paths(self, root: Path) -> list[Path]:
        return sorted(root.rglob("*"), key=lambda candidate: (len(candidate.parts), str(candidate)))


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
