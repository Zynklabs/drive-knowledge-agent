"""Build a LangChain ``Embeddings`` instance from config."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from drive_agent.core.config import ConfigError, Settings


def _require(module: str, extra: str) -> None:
    try:
        __import__(module)
    except ImportError as exc:
        raise ConfigError(
            f"embeddings provider needs an optional extra: "
            f"pip install 'drive-knowledge-agent[{extra}]'"
        ) from exc


def build_embeddings(settings: Settings) -> Embeddings:
    config, secrets = settings.embeddings, settings.secrets
    model = config.model or ""
    match config.provider:
        case "local":
            from drive_agent.embeddings.local import FastEmbedEmbeddings

            return FastEmbedEmbeddings(model, cache_dir=settings.data_dir / "models")
        case "voyage":
            _require("langchain_voyageai", "voyage")
            from langchain_voyageai import VoyageAIEmbeddings

            if secrets.voyage_api_key is None:
                raise ConfigError("embeddings.provider=voyage needs VOYAGE_API_KEY")
            return VoyageAIEmbeddings(model=model, voyage_api_key=secrets.voyage_api_key)
        case "openai":
            _require("langchain_openai", "openai")
            from langchain_openai import OpenAIEmbeddings

            if secrets.openai_api_key is None:
                raise ConfigError("embeddings.provider=openai needs OPENAI_API_KEY")
            return OpenAIEmbeddings(model=model, openai_api_key=secrets.openai_api_key)
        case _:
            raise ConfigError(
                f"embeddings.provider={config.provider} is not supported yet; "
                "use local, voyage, or openai"
            )


__all__ = ["build_embeddings"]
