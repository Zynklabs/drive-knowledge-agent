"""Layered configuration.

Precedence, lowest to highest:

1. Defaults defined on the models below.
2. YAML file: ``--config`` path, else ``$DRIVE_AGENT_CONFIG``,
   else ``./drive-agent.yaml`` if present.
3. Environment (``.env`` file, then the real process environment):
   - friendly aliases from the spec, e.g. ``VECTOR_STORE=qdrant`` (see ``ENV_ALIASES``);
   - any field via ``DRIVE_AGENT__<SECTION>__<FIELD>``,
     e.g. ``DRIVE_AGENT__INGEST__CHUNK_TOKENS=500``.
4. Explicit overrides passed by the caller (CLI flags), as dotted paths.

Secrets (API keys, tokens, DSNs) are read only from the environment, never from YAML.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_CONFIG_FILE = Path("drive-agent.yaml")
CONFIG_ENV_VAR = "DRIVE_AGENT_CONFIG"
NESTED_ENV_PREFIX = "DRIVE_AGENT__"

AuthMode = Literal["service_account", "oauth"]
FileKind = Literal["docs", "slides", "pdf", "docx", "pptx", "txt", "md", "html", "json"]
EmbeddingsProvider = Literal["voyage", "openai", "cohere", "google", "local"]
VectorStoreProvider = Literal["chroma", "qdrant", "pinecone", "pgvector"]

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "voyage": "voyage-3.5",
    "openai": "text-embedding-3-small",
    "cohere": "embed-v4.0",
    "google": "gemini-embedding-001",
    "local": "nomic-ai/nomic-embed-text-v1.5",  # Apache-2.0, 8192-token context, via FastEmbed
}

# Spec-friendly environment variables -> dotted config path.
# When two aliases target the same path, the later entry wins.
ENV_ALIASES: dict[str, str] = {
    "GOOGLE_AUTH_MODE": "drive.auth_mode",
    "GOOGLE_APPLICATION_CREDENTIALS": "drive.credentials_file",
    "GOOGLE_OAUTH_CLIENT_FILE": "drive.oauth_client_file",
    "GOOGLE_IMPERSONATE_USER": "drive.impersonate_user",
    "DRIVE_FOLDER_ID": "drive.folder_ids",
    "DRIVE_FOLDER_IDS": "drive.folder_ids",
    "CONTEXTUAL_CHUNKS": "ingest.contextual_chunks",
    "EMBEDDINGS_PROVIDER": "embeddings.provider",
    "EMBEDDINGS_MODEL": "embeddings.model",
    "VECTOR_STORE": "vector_store.provider",
    "VECTOR_STORE_URL": "vector_store.url",
    "QDRANT_URL": "vector_store.url",
    "LLM_PROVIDER": "llm.provider",
    "ANSWER_MODEL": "llm.answer_model",
    "FAST_MODEL": "llm.fast_model",
    "AGENT_MODE": "agent.mode",
    "TRACING_PROVIDER": "tracing.provider",
    "SLACK_MODE": "interfaces.slack.mode",
}

DEFAULT_FILE_KINDS: tuple[FileKind, ...] = ("docs", "slides", "pdf", "docx", "pptx", "txt", "md")

# Excel, Google Sheets, and CSV are out of scope: tabular data can be huge and chunks poorly.
EXCLUDED_FILE_KINDS = ("xlsx", "xls", "xlsm", "sheets", "csv")


class ConfigError(Exception):
    """Configuration could not be loaded or is invalid."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _split_csv(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


_FOLDER_PATH_RE = re.compile(r"/folders/([A-Za-z0-9_-]+)")
_ID_PARAM_RE = re.compile(r"[?&]id=([A-Za-z0-9_-]+)")
_BARE_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


def extract_folder_id(value: str) -> str:
    """Return the folder ID from a Drive folder URL, an ``?id=`` link, or a bare ID."""
    value = value.strip()
    for pattern in (_FOLDER_PATH_RE, _ID_PARAM_RE):
        if match := pattern.search(value):
            return match.group(1)
    if _BARE_ID_RE.fullmatch(value):
        return value
    raise ValueError(f"not a Google Drive folder URL or ID: {value!r}")


class DriveSettings(_Section):
    auth_mode: AuthMode = "service_account"
    folder_ids: list[str] = Field(default_factory=list)
    credentials_file: Path | None = None
    oauth_client_file: Path | None = None
    impersonate_user: str | None = None
    include_mime: list[FileKind] = Field(default_factory=lambda: list(DEFAULT_FILE_KINDS))
    max_file_mb: int = Field(default=50, gt=0)
    max_depth: int | None = Field(default=None, ge=0)

    @field_validator("include_mime", mode="before")
    @classmethod
    def _parse_include(cls, value: Any) -> Any:
        value = _split_csv(value)
        if isinstance(value, list):
            excluded = sorted({v for v in value if v in EXCLUDED_FILE_KINDS})
            if excluded:
                raise ValueError(
                    f"{', '.join(excluded)} not supported: spreadsheets and tabular files "
                    "are out of scope"
                )
        return value

    @field_validator("folder_ids", mode="before")
    @classmethod
    def _parse_folders(cls, value: Any) -> Any:
        value = _split_csv(value)
        if not isinstance(value, list):
            return value
        ids = [extract_folder_id(str(v)) for v in value]
        return list(dict.fromkeys(ids))


class IngestSettings(_Section):
    chunk_tokens: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    contextual_chunks: bool = False  # Claude-written chunk context; not implemented yet
    schedule_cron: str | None = "0 */6 * * *"

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        return self


class EmbeddingsSettings(_Section):
    provider: EmbeddingsProvider = "local"
    model: str | None = None

    @model_validator(mode="after")
    def _default_model(self) -> Self:
        if self.model is None:
            self.model = DEFAULT_EMBEDDING_MODELS[self.provider]
        return self


class VectorStoreSettings(_Section):
    provider: VectorStoreProvider = "qdrant"
    collection: str = "drive_agent"
    url: str | None = None
    """Server URL, e.g. http://localhost:6333 or a Qdrant Cloud URL. ``:memory:`` for tests.
    When unset, Qdrant runs embedded on disk at ``persist_dir``."""
    persist_dir: Path | None = None
    """Embedded-mode storage; defaults to ``<data_dir>/qdrant``."""


class RetrievalSettings(_Section):
    top_k: int = Field(default=20, gt=0)
    """Candidates retrieved per question (then graded by the fast model)."""
    final_k: int = Field(default=8, gt=0)
    """Most sources passed to the answer model."""
    hybrid: bool = True
    """Dense vectors + BM25 keyword vectors fused with RRF. Changing it needs `sync --full`."""
    grade: bool = True
    """Let the fast model drop irrelevant chunks and retry with a reworded query once."""

    @model_validator(mode="after")
    def _final_within_top(self) -> Self:
        if self.final_k > self.top_k:
            raise ValueError("final_k must not exceed top_k")
        return self


Effort = Literal["low", "medium", "high", "xhigh", "max"]


class LLMSettings(_Section):
    provider: Literal["anthropic", "openrouter"] = "anthropic"
    """``openrouter`` sends the same Anthropic-format requests to OpenRouter."""
    answer_model: str = "claude-opus-5-5"
    answer_effort: Effort = "medium"
    fast_model: str = "claude-haiku-5-5"
    """Query rewriting and relevance grading."""
    fast_effort: Effort = "low"


class AgentSettings(_Section):
    mode: Literal["rag", "agentic", "auto"] = "rag"  # agentic/auto: planned (M7)
    checkpointer: Literal["memory", "sqlite", "postgres"] = "sqlite"


class TracingSettings(_Section):
    provider: Literal["none", "langsmith", "langfuse"] = "none"


class SlackSettings(_Section):
    mode: Literal["socket", "http"] = "socket"
    allowed_channels: list[str] = Field(default_factory=list)

    _split_channels = field_validator("allowed_channels", mode="before")(_split_csv)


class InterfacesSettings(_Section):
    slack: SlackSettings = Field(default_factory=SlackSettings)


class Secrets(BaseModel):
    """Credentials read from the environment only. Field names match env var names."""

    model_config = ConfigDict(extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None
    qdrant_api_key: SecretStr | None = None
    database_url: SecretStr | None = None
    slack_bot_token: SecretStr | None = None
    slack_app_token: SecretStr | None = None
    slack_signing_secret: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: Any) -> Any:
        return None if value == "" else value

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Secrets:
        fields = cls.model_fields
        return cls.model_validate(
            {key.lower(): val for key, val in env.items() if key.lower() in fields}
        )

    def status(self) -> dict[str, bool]:
        """Env var name -> whether it is set. Never exposes values."""
        return {name.upper(): getattr(self, name) is not None for name in type(self).model_fields}


class Settings(_Section):
    data_dir: Path = Path(".drive-agent")
    drive: DriveSettings = Field(default_factory=DriveSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    interfaces: InterfacesSettings = Field(default_factory=InterfacesSettings)
    secrets: Secrets = Field(default_factory=Secrets, exclude=True)
    source_file: Path | None = Field(default=None, exclude=True)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Non-secret settings as plain data, suitable for writing a YAML config.

        A provider-default embedding model is written as null so that changing only the
        provider later picks up that provider's default instead of a mismatched model.
        """
        data = self.model_dump(mode="json")
        if self.embeddings.model == DEFAULT_EMBEDDING_MODELS[self.embeddings.provider]:
            data["embeddings"]["model"] = None
        return data


def _set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = data
    for key in parents:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[leaf] = value


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_env(env_file: Path | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        env.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    env.update(os.environ)
    return env


def _env_layer(env: Mapping[str, str]) -> dict[str, Any]:
    layer: dict[str, Any] = {}
    for name, path in ENV_ALIASES.items():
        if env.get(name):
            _set_path(layer, path, env[name])
    for name, value in env.items():
        if name.startswith(NESTED_ENV_PREFIX) and value != "":
            path = name.removeprefix(NESTED_ENV_PREFIX).lower().replace("__", ".")
            if path == "secrets" or path.startswith("secrets."):
                raise ConfigError(
                    f"{name}: set secrets via their own env vars, e.g. ANTHROPIC_API_KEY"
                )
            _set_path(layer, path, value)
    return layer


def _resolve_config_file(config_file: Path | None, env: Mapping[str, str]) -> Path | None:
    if config_file is not None:
        if not config_file.is_file():
            raise ConfigError(f"config file not found: {config_file}")
        return config_file
    if env.get(CONFIG_ENV_VAR):
        path = Path(env[CONFIG_ENV_VAR])
        if not path.is_file():
            raise ConfigError(f"{CONFIG_ENV_VAR} points to a missing file: {path}")
        return path
    return DEFAULT_CONFIG_FILE if DEFAULT_CONFIG_FILE.is_file() else None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    if "secrets" in data:
        raise ConfigError(
            f"{path}: secrets must not be stored in the config file; "
            "use environment variables or a .env file"
        )
    return data


def load_settings(
    config_file: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    env_file: Path | None = Path(".env"),
) -> Settings:
    """Build ``Settings`` from all layers.

    ``env``, when given, replaces ``.env`` + ``os.environ`` entirely (used by tests and ``init``).
    """
    environment = dict(env) if env is not None else _read_env(env_file)
    source = _resolve_config_file(config_file, environment)

    data: dict[str, Any] = _read_yaml(source) if source else {}
    data = _deep_merge(data, _env_layer(environment))
    for dotted, value in (overrides or {}).items():
        _set_path(data, dotted, value)

    try:
        settings = Settings.model_validate(data)
        secrets = Secrets.from_env(environment)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
    return settings.model_copy(update={"secrets": secrets, "source_file": source})


def _format_validation_error(exc: ValidationError) -> str:
    lines = [f"{exc.error_count()} invalid setting(s):"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)
