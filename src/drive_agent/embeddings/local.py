"""Open-source embeddings run locally with FastEmbed (ONNX, CPU, no PyTorch). Text never leaves
the machine; the model is downloaded once from Hugging Face into ``<data_dir>/models``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

# Some models are trained with task prefixes; FastEmbed does not add them for us.
# (document prefix, query prefix)
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-ai/nomic-embed-text-v1": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1.5-Q": ("search_document: ", "search_query: "),
    "Qwen/Qwen3-Embedding-0.6B": (
        "",
        "Instruct: Given a question, retrieve passages that answer it\nQuery: ",
    ),
    "Qwen/Qwen3-Embedding-0.6B-Q": (
        "",
        "Instruct: Given a question, retrieve passages that answer it\nQuery: ",
    ),
}


MAX_INPUT_TOKENS = 1024
"""Longest input the model sees. Chunks aim for ~800 tokens, but text that tokenizes badly
(garbled PDF text, symbols, other scripts) can be several times longer, and a batch is padded to
its longest input. Attention memory grows with the square of that length, so a model that
accepts 8192 tokens (nomic) can need tens of GB for one batch. Truncating costs only the tail of
an oversized chunk, which the next chunk's overlap partly covers."""


def _load_model(model_name: str, cache_dir: Path) -> Any:
    from fastembed import TextEmbedding  # deferred: imports onnxruntime

    return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))


def _cap_input_tokens(model: Any, max_tokens: int) -> None:
    """Lower the tokenizer's truncation limit to ``max_tokens`` (never raise it)."""
    tokenizer = getattr(getattr(model, "model", None), "tokenizer", None)
    if tokenizer is None:
        return
    truncation = tokenizer.truncation or {}
    if truncation.get("max_length", max_tokens + 1) > max_tokens:
        tokenizer.enable_truncation(
            max_length=max_tokens, direction=truncation.get("direction", "right")
        )


ARENA_SHRINKAGE = ("memory.enable_memory_arena_shrinkage", "cpu:0")
"""onnxruntime run option that hands the CPU memory arena back to the OS after each run. The
arena otherwise keeps its peak for the life of the process: a few full-length batches leave a
sync, or the Slack bot that runs auto-sync, holding ~8 GB long after embedding is done.
Re-allocating per run costs no measurable throughput next to the model's own compute."""


def _release_arena_after_each_run(model: Any) -> None:
    """Make every ``InferenceSession.run`` of ``model`` shrink the arena when it finishes.

    FastEmbed 0.9 exposes session options (``enable_cpu_mem_arena``) but not run options, so we
    wrap the session's public ``run`` and pass ours whenever the caller passed none."""
    session: Any = getattr(getattr(model, "model", None), "model", None)
    run = getattr(session, "run", None)
    if run is None:
        return
    import onnxruntime as ort

    options = ort.RunOptions()
    options.add_run_config_entry(*ARENA_SHRINKAGE)

    def run_and_shrink(output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
        return run(output_names, input_feed, run_options or options)

    session.run = run_and_shrink


BATCH_SIZE = 4
"""Texts per model run. Peak memory grows with batch size times the square of the batch's
longest input: 16 inputs at the 1,024-token cap peaked near 8 GB, 4 stay under 3 GB. Smaller
batches also pad less, so with length-sorted input they are faster, not slower."""


class FastEmbedEmbeddings(Embeddings):
    """LangChain ``Embeddings`` over a FastEmbed ``TextEmbedding`` model."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        *,
        batch_size: int = BATCH_SIZE,
        max_tokens: int = MAX_INPUT_TOKENS,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = _load_model(model_name, cache_dir)
        _cap_input_tokens(self._model, max_tokens)
        _release_arena_after_each_run(self._model)
        self._batch_size = batch_size
        self._doc_prefix, self._query_prefix = MODEL_PREFIXES.get(model_name, ("", ""))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed longest first: a batch is padded to its longest input, so grouping similar
        lengths cuts wasted compute, and the most memory-hungry batch runs first, not mid-sync."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        vectors = self._model.embed(
            [self._doc_prefix + texts[i] for i in order], batch_size=self._batch_size
        )
        result: list[list[float]] = [[] for _ in texts]
        for i, vector in zip(order, vectors, strict=True):
            result[i] = vector.tolist()
        return result

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([self._query_prefix + text])))
        result: list[float] = vector.tolist()
        return result
