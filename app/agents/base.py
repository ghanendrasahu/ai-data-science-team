"""Base helpers shared by every agent node."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.services.llm import LLMService


class BaseAgent(ABC):
    """Common interface for LangGraph node agents.

    Subclasses implement :meth:`run` which returns a partial state update.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.llm = LLMService.instance()

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        update = self.run(state)
        update.setdefault("status", f"{self.name}_done")
        return update

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a partial state update."""
