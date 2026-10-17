"""Offline stand-ins for Claude and the BM25 encoder."""

from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_qdrant import SparseEmbeddings, SparseVector
from pydantic import BaseModel

from drive_agent.graph.rag import Relevant, SearchQuery
from drive_agent.llm import Answer

T = TypeVar("T", bound=BaseModel)


class FakeSparse(SparseEmbeddings):
    """Bag-of-words sparse vectors (hashed token counts), deterministic."""

    def _vector(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in text.lower().split():
            index = zlib.crc32(token.encode()) % 100_000
            counts[index] = counts.get(index, 0.0) + 1.0
        indices = sorted(counts)
        return SparseVector(indices=indices, values=[counts[i] for i in indices])

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._vector(text)


class FakeLLM:
    """Scripted Claude: queues for structured outputs, a fixed streamed answer."""

    def __init__(
        self,
        *,
        answer: str = "Jane knows Go [1].",
        grades: list[list[int]] | None = None,
        rewrites: list[str] | None = None,
        refuse: bool = False,
    ) -> None:
        self.answer_text = answer
        self.grades = list(grades) if grades is not None else []
        self.rewrites = list(rewrites) if rewrites is not None else []
        self.refuse = refuse
        self.structured_calls: list[tuple[str, str, str]] = []
        self.answer_calls: list[tuple[str, list[dict[str, Any]]]] = []

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        self.structured_calls.append((schema.__name__, system, prompt))
        if schema is Relevant:
            passages = self.grades.pop(0) if self.grades else [1, 2, 3, 4, 5, 6, 7, 8]
            return schema.model_validate({"passages": passages})
        if schema is SearchQuery:
            query = self.rewrites.pop(0) if self.rewrites else "rewritten query"
            return schema.model_validate({"query": query})
        raise AssertionError(f"unexpected schema {schema}")

    def answer(
        self,
        system: str,
        messages: list[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        self.answer_calls.append((system, messages))
        if self.refuse:
            return Answer(text="", model="fake", refused=True)
        for word in self.answer_text.split(" "):
            if on_text is not None:
                on_text(word + " ")
        return Answer(text=self.answer_text, model="fake")
