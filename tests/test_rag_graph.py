from typing import Any

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver

from drive_agent.core.agent import DriveAgent
from drive_agent.core.config import ConfigError, Settings
from drive_agent.graph import prompts
from drive_agent.graph.rag import parse_citations

from .fakes import FakeLLM

DOCS = [
    Document(
        "Jane.pdf > page 1\n\nJane Doe. Go, Kubernetes.",
        metadata={
            "file_name": "Jane.pdf",
            "path": "H/Jane.pdf",
            "web_url": "https://d/1",
            "page": 1,
        },
    ),
    Document(
        "Raj.pdf > page 2\n\nRaj. React, TypeScript.",
        metadata={"file_name": "Raj.pdf", "path": "H/Raj.pdf", "web_url": "https://d/2", "page": 2},
    ),
    Document(
        "Deck > slide 3\n\nHiring plan",
        metadata={"file_name": "Deck", "slide": 3, "section": "Plan"},
    ),
]


class Retriever:
    def __init__(self, results: dict[str, list[Document]] | None = None) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int) -> list[Document]:
        self.calls.append((query, k))
        if self.results is None:
            return DOCS[:k]
        return self.results.get(query, [])


def agent(llm: FakeLLM, retriever: Retriever, **retrieval: Any) -> DriveAgent:
    settings = Settings.model_validate({"retrieval": retrieval} if retrieval else {})
    return DriveAgent(settings, llm, retriever, InMemorySaver())


def test_first_question_streams_answer_with_citations() -> None:
    llm, retriever = FakeLLM(answer="Jane knows Go [1]."), Retriever()
    tokens: list[str] = []
    reply = agent(llm, retriever).ask("Who knows Go?", thread_id="t1", on_text=tokens.append)

    assert reply.status == "answered"
    assert reply.answer == "Jane knows Go [1]."
    assert "".join(tokens).strip() == "Jane knows Go [1]."
    assert reply.citations == [
        {
            "number": 1,
            "file_name": "Jane.pdf",
            "path": "H/Jane.pdf",
            "web_url": "https://d/1",
            "page": 1,
        }
    ]
    assert reply.query == "Who knows Go?"  # no history: no rewrite call
    assert retriever.calls == [("Who knows Go?", 20)]
    assert [c[0] for c in llm.structured_calls] == ["Relevant"]

    system, messages = llm.answer_calls[0]
    assert system == prompts.ANSWER_SYSTEM
    assert messages[-1]["role"] == "user"
    content = messages[-1]["content"]
    assert '<source id="1">\nJane.pdf > page 1' in content
    assert content.endswith("Question: Who knows Go?")


def test_grading_keeps_relevant_passages_in_rank_order() -> None:
    llm = FakeLLM(grades=[[3, 1]], answer="See [2] and [1].")
    reply = agent(llm, Retriever()).ask("q", thread_id="t")
    sources = llm.answer_calls[0][1][-1]["content"]
    assert sources.index("Jane.pdf") < sources.index("Deck")
    assert "Raj.pdf" not in sources
    assert [c["file_name"] for c in reply.citations] == ["Deck", "Jane.pdf"]


def test_follow_up_uses_conversation_memory() -> None:
    llm = FakeLLM(rewrites=["Jane Doe Kubernetes experience"])
    retriever = Retriever()
    bot = agent(llm, retriever)
    bot.ask("Who knows Go?", thread_id="t1")
    reply = bot.ask("Does she know Kubernetes?", thread_id="t1")

    assert reply.query == "Jane Doe Kubernetes experience"
    assert retriever.calls[-1][0] == "Jane Doe Kubernetes experience"
    rewrite = next(c for c in llm.structured_calls if c[0] == "SearchQuery")
    assert "user: Who knows Go?" in rewrite[2]
    assert "Latest question: Does she know Kubernetes?" in rewrite[2]
    messages = llm.answer_calls[-1][1]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "Who knows Go?"


def test_threads_are_isolated() -> None:
    llm = FakeLLM()
    bot = agent(llm, Retriever())
    bot.ask("Who knows Go?", thread_id="a")
    reply = bot.ask("Who knows React?", thread_id="b")
    assert reply.query == "Who knows React?"  # no rewrite: thread b has no history
    assert [m["role"] for m in llm.answer_calls[-1][1]] == ["user"]


def test_nothing_relevant_retries_with_reformulated_query() -> None:
    llm = FakeLLM(grades=[[], [1]], rewrites=["engineering manager led team"])
    retriever = Retriever({"leadership?": DOCS[1:], "engineering manager led team": DOCS[:1]})
    reply = agent(llm, retriever).ask("leadership?", thread_id="t")
    assert [q for q, _ in retriever.calls] == ["leadership?", "engineering manager led team"]
    assert reply.status == "answered"
    assert [c["file_name"] for c in reply.citations] == ["Jane.pdf"]


def test_still_nothing_relevant_answers_not_found_without_claude() -> None:
    llm = FakeLLM(grades=[[], []])
    tokens: list[str] = []
    reply = agent(llm, Retriever()).ask("unrelated", thread_id="t", on_text=tokens.append)
    assert reply.status == "not_found"
    assert reply.answer == prompts.NOT_FOUND
    assert "".join(tokens) == prompts.NOT_FOUND
    assert reply.citations == []
    assert llm.answer_calls == []


def test_empty_index_results_skip_grading() -> None:
    llm = FakeLLM(rewrites=["other"])
    reply = agent(llm, Retriever({})).ask("anything", thread_id="t")
    assert reply.status == "not_found"
    assert [c[0] for c in llm.structured_calls] == ["SearchQuery"]  # only the reformulation


def test_grading_disabled() -> None:
    llm = FakeLLM(answer="[3]")
    reply = agent(llm, Retriever(), grade=False, final_k=2, top_k=5).ask("q", thread_id="t")
    assert llm.structured_calls == []
    assert reply.citations == []  # [3] is out of range: only 2 sources were kept


def test_refusal() -> None:
    reply = agent(FakeLLM(refuse=True), Retriever()).ask("q", thread_id="t")
    assert reply.status == "refused"
    assert reply.answer == prompts.REFUSED


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A [2] and B [1][2].", [2, 1]),
        ("Both [1, 3].", [1, 3]),
        ("Bad [9] and [0].", []),
        ("No citations.", []),
    ],
)
def test_parse_citations(answer: str, expected: list[int]) -> None:
    sources: list[Any] = [{"text": d.page_content, "metadata": d.metadata} for d in DOCS]
    assert [c["number"] for c in parse_citations(answer, sources)] == expected


def test_unimplemented_agent_modes() -> None:
    settings = Settings.model_validate({"agent": {"mode": "agentic"}})
    with pytest.raises(ConfigError, match="planned for M7"):
        DriveAgent(settings, FakeLLM(), Retriever(), InMemorySaver())


def test_system_prompt_guards_against_injection() -> None:
    assert "not instructions" in prompts.ANSWER_SYSTEM
    assert "[2]" in prompts.ANSWER_SYSTEM
