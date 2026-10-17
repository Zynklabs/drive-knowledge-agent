"""ClaudeLLM against a stubbed SDK client: checks the exact request shape we send."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, SecretStr

from drive_agent.core.config import Settings
from drive_agent.llm.claude import FALLBACK_BETA, MISSING_CREDENTIALS, ClaudeLLM, LLMError


class Out(BaseModel):
    query: str


class StubClient:
    def __init__(
        self, *, parse_result: Any = None, final: Any = None, error: Exception | None = None
    ):
        self.parse_kwargs: dict[str, Any] = {}
        self.stream_kwargs: dict[str, Any] = {}
        self._parse_result, self._final, self._error = parse_result, final, error
        self.messages = SimpleNamespace(parse=self._parse)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _parse(self, **kwargs: Any) -> Any:
        self.parse_kwargs = kwargs
        if self._error:
            raise self._error
        return self._parse_result

    @contextmanager
    def _stream(self, **kwargs: Any) -> Iterator[Any]:
        self.stream_kwargs = kwargs
        if self._error:
            raise self._error
        yield SimpleNamespace(
            text_stream=iter(["Jane ", "knows ", "Go [1]."]),
            get_final_message=lambda: self._final,
        )


def text_block(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def llm(client: StubClient, **llm_config: Any) -> ClaudeLLM:
    return ClaudeLLM(Settings.model_validate({"llm": llm_config}), client=client)  # type: ignore[arg-type]


def test_structured_uses_fast_model_low_effort_and_schema() -> None:
    client = StubClient(
        parse_result=SimpleNamespace(stop_reason="end_turn", parsed_output=Out(query="q"))
    )
    result = llm(client).structured("sys", "prompt", Out)
    assert result == Out(query="q")
    kw = client.parse_kwargs
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["output_config"] == {"effort": "low"}
    assert kw["output_format"] is Out
    assert kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "prompt"}]
    assert "fallbacks" not in kw  # Haiku has no server-side fallback


def test_structured_refusal_is_an_error() -> None:
    client = StubClient(parse_result=SimpleNamespace(stop_reason="refusal", parsed_output=None))
    with pytest.raises(LLMError, match="no usable output"):
        llm(client).structured("s", "p", Out)


def test_answer_streams_with_opus_effort_fallback_and_cached_system() -> None:
    final = SimpleNamespace(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            text_block("Jane knows Go [1]."),
        ],
    )
    client = StubClient(final=final)
    seen: list[str] = []
    answer = llm(client).answer("SYSTEM", [{"role": "user", "content": "q"}], on_text=seen.append)

    assert answer.text == "Jane knows Go [1]."
    assert answer.refused is False
    assert "".join(seen) == "Jane knows Go [1]."
    kw = client.stream_kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["betas"] == [FALLBACK_BETA]
    assert kw["fallbacks"] == "default"
    assert kw["system"] == [
        {"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}
    ]
    assert "thinking" not in kw  # Opus 5.5: adaptive thinking is implicit
    assert "temperature" not in kw  # sampling parameters are rejected on Opus 5.5


def test_answer_respects_configured_model_and_effort() -> None:
    final = SimpleNamespace(stop_reason="end_turn", model="m", content=[text_block("x")])
    client = StubClient(final=final)
    llm(client, answer_model="claude-sonnet-5-5", answer_effort="high").answer("s", [])
    assert client.stream_kwargs["model"] == "claude-sonnet-5-5"
    assert client.stream_kwargs["output_config"] == {"effort": "high"}


def test_answer_refusal() -> None:
    final = SimpleNamespace(stop_reason="refusal", model="m", content=[])
    assert llm(StubClient(final=final)).answer("s", []).refused is True


def test_missing_credentials_message() -> None:
    error = TypeError("Could not resolve authentication method. Expected one of api_key ...")
    with pytest.raises(LLMError) as info:
        llm(StubClient(error=error)).answer("s", [])
    assert str(info.value) == MISSING_CREDENTIALS


def test_other_type_errors_are_not_masked() -> None:
    with pytest.raises(TypeError, match="boom"):
        llm(StubClient(error=TypeError("boom"))).structured("s", "p", Out)


def test_connection_error_is_explained() -> None:
    error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com")
    )
    with pytest.raises(LLMError, match="Could not reach the Claude API"):
        llm(StubClient(error=error)).answer("s", [])


def test_api_key_from_settings_is_used() -> None:
    settings = Settings()
    settings.secrets = settings.secrets.model_copy(
        update={"anthropic_api_key": SecretStr("sk-test")}
    )
    client = ClaudeLLM(settings)._client
    assert client.api_key == "sk-test"
