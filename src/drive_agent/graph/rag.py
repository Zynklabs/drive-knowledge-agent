"""The RAG graph (LangGraph).

    START → rewrite → retrieve → grade ─┬─ relevant ─────────→ generate → cite → END
                         ↑              ├─ none, first try ─→ reformulate ─┘ (back to retrieve)
                         └──────────────┘
                                        └─ none again ──────→ not_found → END

State is checkpointed per ``thread_id``, so follow-up questions see the conversation.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from drive_agent.core.config import RetrievalSettings
from drive_agent.graph import prompts
from drive_agent.llm import LLM

Retriever = Callable[[str, int], list[Document]]
MAX_HISTORY_TURNS = 10
MAX_ATTEMPTS = 2  # first search + one reformulated retry
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class Turn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Source(TypedDict):
    text: str
    metadata: dict[str, Any]


class Citation(TypedDict, total=False):
    number: int
    file_name: str
    path: str
    web_url: str
    page: int
    slide: int
    section: str


class RAGState(TypedDict, total=False):
    question: str
    history: Annotated[list[Turn], operator.add]
    query: str
    attempts: int
    sources: list[Source]
    answer: str
    status: Literal["answered", "not_found", "refused"]
    citations: list[Citation]


class SearchQuery(BaseModel):
    query: str = Field(description="The search query")


class Relevant(BaseModel):
    passages: list[int] = Field(description="Numbers of the passages that help answer")


def _emit(text: str) -> None:
    get_stream_writer()({"type": "token", "text": text})


def _format_history(history: list[Turn]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)


def _format_sources(sources: list[Source]) -> str:
    return "\n\n".join(
        f'<source id="{n}">\n{source["text"]}\n</source>' for n, source in enumerate(sources, 1)
    )


def parse_citations(answer: str, sources: list[Source]) -> list[Citation]:
    """Map ``[n]`` markers in the answer to source metadata, in order of first use."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        for number in (int(n) for n in match.group(1).split(",")):
            if 1 <= number <= len(sources) and number not in seen:
                seen.append(number)
    citations: list[Citation] = []
    for number in seen:
        meta = sources[number - 1]["metadata"]
        citation: Citation = {"number": number}
        for key in ("file_name", "path", "web_url", "page", "slide", "section"):
            if key in meta:
                citation[key] = meta[key]
        citations.append(citation)
    return citations


def build_rag_graph(
    llm: LLM,
    retriever: Retriever,
    retrieval: RetrievalSettings,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def rewrite(state: RAGState) -> dict[str, Any]:
        history = state.get("history", [])[-2 * MAX_HISTORY_TURNS :]
        question = state["question"]
        if not history:
            return {"query": question, "attempts": 0}
        prompt = (
            f"<conversation>\n{_format_history(history)}\n</conversation>\n\n"
            f"Latest question: {question}"
        )
        return {
            "query": llm.structured(prompts.REWRITE_SYSTEM, prompt, SearchQuery).query,
            "attempts": 0,
        }

    def retrieve(state: RAGState) -> dict[str, Any]:
        docs = retriever(state["query"], retrieval.top_k)
        return {"sources": [{"text": d.page_content, "metadata": dict(d.metadata)} for d in docs]}

    def grade(state: RAGState) -> dict[str, Any]:
        sources = state.get("sources", [])
        attempts = state.get("attempts", 0) + 1
        if not sources or not retrieval.grade:
            return {"sources": sources[: retrieval.final_k], "attempts": attempts}
        prompt = f"Question: {state['question']}\n\n" + "\n\n".join(
            f'<passage n="{n}">\n{s["text"]}\n</passage>' for n, s in enumerate(sources, 1)
        )
        keep = set(llm.structured(prompts.GRADE_SYSTEM, prompt, Relevant).passages)
        kept = [s for n, s in enumerate(sources, 1) if n in keep]
        return {"sources": kept[: retrieval.final_k], "attempts": attempts}

    def after_grade(state: RAGState) -> str:
        if state.get("sources"):
            return "generate"
        if retrieval.grade and state.get("attempts", 0) < MAX_ATTEMPTS:
            return "reformulate"
        return "not_found"

    def reformulate(state: RAGState) -> dict[str, Any]:
        prompt = f"Question: {state['question']}\nSearch query that found nothing: {state['query']}"
        return {"query": llm.structured(prompts.REFORMULATE_SYSTEM, prompt, SearchQuery).query}

    def generate(state: RAGState) -> dict[str, Any]:
        history = state.get("history", [])[-2 * MAX_HISTORY_TURNS :]
        messages: list[dict[str, Any]] = [dict(turn) for turn in history]
        messages.append(
            {
                "role": "user",
                "content": f"<sources>\n{_format_sources(state['sources'])}\n</sources>\n\n"
                f"Question: {state['question']}",
            }
        )
        result = llm.answer(prompts.ANSWER_SYSTEM, messages, on_text=_emit)
        if result.refused:
            _emit(prompts.REFUSED)
            return {"answer": prompts.REFUSED, "status": "refused"}
        return {"answer": result.text, "status": "answered"}

    def not_found(state: RAGState) -> dict[str, Any]:
        _emit(prompts.NOT_FOUND)
        return {"answer": prompts.NOT_FOUND, "status": "not_found", "sources": []}

    def cite(state: RAGState) -> dict[str, Any]:
        answer = state["answer"]
        return {
            "citations": parse_citations(answer, state.get("sources", [])),
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ],
        }

    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("reformulate", reformulate)
    graph.add_node("generate", generate)
    graph.add_node("not_found", not_found)
    graph.add_node("cite", cite)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["generate", "reformulate", "not_found"])
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", "cite")
    graph.add_edge("not_found", "cite")
    graph.add_edge("cite", END)
    return graph.compile(checkpointer=checkpointer)
