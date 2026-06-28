from __future__ import annotations

from abc import ABC, abstractmethod

from harnessgym.config import RunConfig
from harnessgym.models import IterationContext, RunnerResult


class Runner(ABC):
    """Common runner interface for HarnessGym Codex backends."""

    @abstractmethod
    def start_attempt(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
    ) -> RunnerResult:
        raise NotImplementedError

    @abstractmethod
    def reflect(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise NotImplementedError

    @abstractmethod
    def build_tooling(
        self,
        config: RunConfig,
        context: IterationContext,
        prompt: str,
        session_id: str | None,
    ) -> RunnerResult:
        raise NotImplementedError

    def close(self) -> None:
        return None
