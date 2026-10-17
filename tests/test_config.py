from pathlib import Path

import pytest
import yaml

from drive_agent.core.config import (
    DEFAULT_EMBEDDING_MODELS,
    ConfigError,
    Settings,
    extract_folder_id,
    load_settings,
)

from .conftest import REPO_ROOT

FOLDER = "1AbCdEfGhIjKlMnOpQrStUvWxYz_-0123"


def write_yaml(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- folder IDs -------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        FOLDER,
        f"  {FOLDER}  ",
        f"https://drive.google.com/drive/folders/{FOLDER}",
        f"https://drive.google.com/drive/u/0/folders/{FOLDER}?usp=sharing",
        f"https://drive.google.com/open?id={FOLDER}",
    ],
)
def test_extract_folder_id(value: str) -> None:
    assert extract_folder_id(value) == FOLDER


@pytest.mark.parametrize("value", ["", "not a folder!", "https://example.com/x y"])
def test_extract_folder_id_rejects_garbage(value: str) -> None:
    with pytest.raises(ValueError, match="not a Google Drive folder"):
        extract_folder_id(value)


# --- defaults & layering ----------------------------------------------------


def test_defaults_match_spec() -> None:
    s = load_settings(env={})
    assert s.source_file is None
    assert s.drive.auth_mode == "service_account"
    assert s.drive.folder_ids == []
    assert s.ingest.chunk_tokens == 800
    assert s.ingest.chunk_overlap == 120
    assert s.ingest.contextual_chunks is False
    assert s.embeddings.provider == "local"
    assert s.embeddings.model == "nomic-ai/nomic-embed-text-v1.5"
    assert s.vector_store.provider == "qdrant"
    assert s.vector_store.url is None
    assert s.retrieval.top_k == 20
    assert s.retrieval.hybrid is True
    assert s.retrieval.final_k == 8
    assert s.llm.answer_model == "claude-opus-5-5"
    assert s.llm.answer_effort == "medium"
    assert s.llm.fast_model == "claude-haiku-5-5"
    assert s.agent.mode == "rag"


def test_yaml_layer(tmp_path: Path) -> None:
    cfg = write_yaml(
        tmp_path / "custom.yaml",
        {"drive": {"folder_ids": [FOLDER]}, "vector_store": {"provider": "qdrant"}},
    )
    s = load_settings(cfg, env={})
    assert s.source_file == cfg
    assert s.drive.folder_ids == [FOLDER]
    assert s.vector_store.provider == "qdrant"
    assert s.ingest.chunk_tokens == 800  # untouched sections keep defaults


def test_default_config_file_in_cwd_is_used() -> None:
    write_yaml(Path("drive-agent.yaml"), {"agent": {"mode": "auto"}})
    s = load_settings(env={})
    assert s.agent.mode == "auto"
    assert s.source_file == Path("drive-agent.yaml")


def test_config_path_from_env(tmp_path: Path) -> None:
    cfg = write_yaml(tmp_path / "elsewhere.yaml", {"agent": {"mode": "agentic"}})
    s = load_settings(env={"DRIVE_AGENT_CONFIG": str(cfg)})
    assert s.agent.mode == "agentic"


def test_env_alias_beats_yaml(tmp_path: Path) -> None:
    cfg = write_yaml(tmp_path / "c.yaml", {"vector_store": {"provider": "qdrant"}})
    s = load_settings(cfg, env={"VECTOR_STORE": "pinecone", "CONTEXTUAL_CHUNKS": "false"})
    assert s.vector_store.provider == "pinecone"
    assert s.ingest.contextual_chunks is False


def test_nested_env_sets_any_field() -> None:
    s = load_settings(
        env={
            "DRIVE_AGENT__INGEST__CHUNK_TOKENS": "500",
            "DRIVE_AGENT__INTERFACES__SLACK__ALLOWED_CHANNELS": "C1, C2",
        }
    )
    assert s.ingest.chunk_tokens == 500
    assert s.interfaces.slack.allowed_channels == ["C1", "C2"]


def test_overrides_beat_env() -> None:
    s = load_settings(env={"AGENT_MODE": "rag"}, overrides={"agent.mode": "agentic"})
    assert s.agent.mode == "agentic"


def test_folder_ids_from_env_are_parsed_and_deduped() -> None:
    url = f"https://drive.google.com/drive/folders/{FOLDER}"
    s = load_settings(env={"DRIVE_FOLDER_IDS": f"{url}, {FOLDER}, other_id"})
    assert s.drive.folder_ids == [FOLDER, "other_id"]


def test_google_credential_paths_from_env() -> None:
    s = load_settings(
        env={"GOOGLE_APPLICATION_CREDENTIALS": "/keys/sa.json", "GOOGLE_AUTH_MODE": "oauth"}
    )
    assert s.drive.credentials_file == Path("/keys/sa.json")
    assert s.drive.auth_mode == "oauth"


def test_dotenv_file_is_read_and_process_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("VECTOR_STORE=qdrant\nAGENT_MODE=rag\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MODE", "agentic")
    s = load_settings(env_file=env_file)
    assert s.vector_store.provider == "qdrant"
    assert s.agent.mode == "agentic"


@pytest.mark.parametrize(("provider", "model"), list(DEFAULT_EMBEDDING_MODELS.items()))
def test_embedding_model_defaults_per_provider(provider: str, model: str) -> None:
    assert load_settings(env={"EMBEDDINGS_PROVIDER": provider}).embeddings.model == model


def test_explicit_embedding_model_is_kept() -> None:
    s = load_settings(env={"EMBEDDINGS_PROVIDER": "openai", "EMBEDDINGS_MODEL": "custom"})
    assert s.embeddings.model == "custom"


# --- validation -------------------------------------------------------------


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml", env={})


def test_invalid_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("drive: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_settings(cfg, env={})


def test_non_mapping_yaml(tmp_path: Path) -> None:
    cfg = write_yaml(tmp_path / "list.yaml", ["a", "b"])
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_settings(cfg, env={})


def test_secrets_in_yaml_rejected(tmp_path: Path) -> None:
    cfg = write_yaml(tmp_path / "c.yaml", {"secrets": {"anthropic_api_key": "sk-x"}})
    with pytest.raises(ConfigError, match="secrets must not be stored"):
        load_settings(cfg, env={})


def test_secrets_via_nested_env_rejected() -> None:
    with pytest.raises(ConfigError, match="set secrets via"):
        load_settings(env={"DRIVE_AGENT__SECRETS__ANTHROPIC_API_KEY": "sk-x"})


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    cfg = write_yaml(tmp_path / "c.yaml", {"drive": {"folder": FOLDER}})
    with pytest.raises(ConfigError, match=r"drive\.folder"):
        load_settings(cfg, env={})


def test_invalid_choice_reports_location() -> None:
    with pytest.raises(ConfigError, match=r"vector_store\.provider"):
        load_settings(env={"VECTOR_STORE": "not-a-db"})


@pytest.mark.parametrize("kind", ["xlsx", "xls", "xlsm", "sheets", "csv"])
def test_spreadsheets_are_out_of_scope(kind: str) -> None:
    with pytest.raises(ConfigError, match="out of scope"):
        load_settings(env={}, overrides={"drive.include_mime": ["docs", kind]})


def test_default_file_types_exclude_spreadsheets() -> None:
    kinds = set(Settings().drive.include_mime)
    assert kinds.isdisjoint({"xlsx", "xls", "xlsm", "sheets", "csv"})


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ConfigError, match="chunk_overlap"):
        load_settings(env={}, overrides={"ingest.chunk_overlap": 800})


def test_final_k_within_top_k() -> None:
    with pytest.raises(ConfigError, match="final_k"):
        load_settings(env={}, overrides={"retrieval.final_k": 50})


# --- secrets ----------------------------------------------------------------


def test_secrets_read_from_env_and_masked() -> None:
    s = load_settings(env={"ANTHROPIC_API_KEY": "sk-ant-123", "VOYAGE_API_KEY": ""})
    assert s.secrets.anthropic_api_key is not None
    assert s.secrets.anthropic_api_key.get_secret_value() == "sk-ant-123"
    assert s.secrets.voyage_api_key is None
    assert "sk-ant-123" not in repr(s)
    status = s.secrets.status()
    assert status["ANTHROPIC_API_KEY"] is True
    assert status["VOYAGE_API_KEY"] is False


def test_dump_leaves_default_embedding_model_unset(tmp_path: Path) -> None:
    dumped = load_settings(env={}).to_yaml_dict()
    assert dumped["embeddings"]["model"] is None
    dumped["embeddings"]["provider"] = "openai"
    reloaded = load_settings(write_yaml(tmp_path / "c.yaml", dumped), env={})
    assert reloaded.embeddings.model == DEFAULT_EMBEDDING_MODELS["openai"]

    custom = load_settings(env={"EMBEDDINGS_MODEL": "voyage-3-large"}).to_yaml_dict()
    assert custom["embeddings"]["model"] == "voyage-3-large"


def test_secrets_never_dumped() -> None:
    s = load_settings(env={"ANTHROPIC_API_KEY": "sk-ant-123"})
    dumped = s.to_yaml_dict()
    assert "secrets" not in dumped
    assert "sk-ant-123" not in yaml.safe_dump(dumped)


# --- shipped example files --------------------------------------------------


def test_example_yaml_loads() -> None:
    s = load_settings(REPO_ROOT / "drive-agent.example.yaml", env={})
    assert s.drive.folder_ids


def test_env_example_loads() -> None:
    s = load_settings(env={}, env_file=None)  # baseline
    from_example = load_settings(env_file=REPO_ROOT / ".env.example")
    assert from_example.drive.auth_mode == s.drive.auth_mode
    assert from_example.vector_store.provider == "qdrant"
    assert not any(from_example.secrets.status().values())


def test_llm_provider_and_openrouter_key_from_env() -> None:
    s = load_settings(env={"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-x"})
    assert s.llm.provider == "openrouter"
    assert s.secrets.openrouter_api_key is not None
    assert s.secrets.status()["OPENROUTER_API_KEY"] is True
