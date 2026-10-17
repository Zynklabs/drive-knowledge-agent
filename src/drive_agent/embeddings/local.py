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


def _load_model(model_name: str, cache_dir: Path) -> Any:
    from fastembed import TextEmbedding  # deferred: imports onnxruntime

    return TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))


class FastEmbedEmbeddings(Embeddings):
    """LangChain ``Embeddings`` over a FastEmbed ``TextEmbedding`` model."""

    def __init__(self, model_name: str, cache_dir: Path, *, batch_size: int = 16) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = _load_model(model_name, cache_dir)
        self._batch_size = batch_size
        self._doc_prefix, self._query_prefix = MODEL_PREFIXES.get(model_name, ("", ""))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._doc_prefix + text for text in texts]
        return [
            vector.tolist() for vector in self._model.embed(prefixed, batch_size=self._batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self._model.embed([self._query_prefix + text])))
        result: list[float] = vector.tolist()
        return result
