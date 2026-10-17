"""LLM interface used by the agent graph; Claude (``llm.claude``) is the reference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from drive_agent.llm.claude import Answer, LLMError

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def structured(self, system: str, prompt: str, schema: type[T]) -> T: ...

    def answer(
        self,
        system: str,
        messages: list[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Answer: ...


__all__ = ["LLM", "Answer", "LLMError"]
