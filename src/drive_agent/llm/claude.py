"""Claude via the official ``anthropic`` SDK, directly or through OpenRouter.

OpenRouter accepts Anthropic-format requests at ``OPENROUTER_BASE_URL``, so the same SDK calls
work with ``llm.provider=openrouter``; only Anthropic's server-side fallback beta is left out.

- Answers: streamed from the answer model (default Claude Opus 5.5) with explicit effort and
  server-side refusal fallback (``fallbacks="default"``).
- Small structured tasks (query rewrite, relevance grading): the fast model (default Claude
  Haiku 5.5) with structured outputs via ``messages.parse``. Haiku has no server-side fallback.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from drive_agent.core.config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
_VERSIONED = re.compile(r"^(claude-[a-z]+-\d+)-(\d+)$")
ANSWER_MAX_TOKENS = 64_000  # streamed, so a generous ceiling costs nothing unless used
FAST_MAX_TOKENS = 4_000

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Claude could not be reached or declined the request."""


@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    refused: bool = False


def model_id(model: str, provider: str) -> str:
    """Anthropic model IDs as the provider expects them.

    OpenRouter names Claude models ``anthropic/claude-opus-5.5``; IDs that already contain a
    ``/`` are passed through unchanged.
    """
    if provider != "openrouter" or "/" in model:
        return model
    return "anthropic/" + _VERSIONED.sub(r"\1.\2", model)


def _make_client(settings: Settings) -> anthropic.Anthropic:
    if settings.llm.provider == "openrouter":
        key = settings.secrets.openrouter_api_key
        if key is None:
            raise LLMError("llm.provider=openrouter needs OPENROUTER_API_KEY in .env")
        return anthropic.Anthropic(
            api_key=None,
            auth_token=key.get_secret_value(),
            base_url=OPENROUTER_BASE_URL,
            default_headers={"X-Title": "Drive Knowledge Agent"},
        )
    key = settings.secrets.anthropic_api_key
    # No key: the SDK falls back to ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
    return anthropic.Anthropic(api_key=key.get_secret_value()) if key else anthropic.Anthropic()


class ClaudeLLM:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self._config = settings.llm
        self._openrouter = settings.llm.provider == "openrouter"
        self._answer_model = model_id(settings.llm.answer_model, settings.llm.provider)
        self._fast_model = model_id(settings.llm.fast_model, settings.llm.provider)
        self._client = client or _make_client(settings)

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        """One fast-model call returning a validated ``schema`` instance."""
        try:
            response = self._client.messages.parse(
                model=self._fast_model,
                max_tokens=FAST_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self._config.fast_effort},
                output_format=schema,
            )
        except anthropic.APIError as exc:
            raise LLMError(_describe(exc)) from exc
        except TypeError as exc:
            raise _maybe_missing_credentials(exc) from exc
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise LLMError(f"{self._fast_model} returned no usable output ({response.stop_reason})")
        return response.parsed_output

    def answer(
        self,
        system: str,
        messages: list[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        """Stream an answer; ``on_text`` receives text deltas as they arrive."""
        # Server-side refusal fallback is an Anthropic API feature; OpenRouter rejects it.
        fallback: dict[str, Any] = (
            {} if self._openrouter else {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        )
        try:
            with self._client.beta.messages.stream(
                model=self._answer_model,
                max_tokens=ANSWER_MAX_TOKENS,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,  # type: ignore[arg-type]
                output_config={"effort": self._config.answer_effort},
                **fallback,
            ) as stream:
                for text in stream.text_stream:
                    if on_text is not None:
                        on_text(text)
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            raise LLMError(_describe(exc)) from exc
        except TypeError as exc:
            raise _maybe_missing_credentials(exc) from exc

        text = "".join(block.text for block in final.content if block.type == "text")
        if final.stop_reason == "refusal":
            return Answer(text="", model=final.model, refused=True)
        return Answer(text=text, model=final.model)


MISSING_CREDENTIALS = (
    "No Claude credentials found. Add ANTHROPIC_API_KEY=... to .env (or the environment), "
    "or sign in with `ant auth login`."
)


def _maybe_missing_credentials(exc: TypeError) -> Exception:
    # The SDK raises TypeError("Could not resolve authentication method ...") at request time.
    if "authentication" in str(exc):
        return LLMError(MISSING_CREDENTIALS)
    return exc


def _describe(exc: anthropic.APIError) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "Claude rejected the credentials: set ANTHROPIC_API_KEY (or run `ant auth login`)."
    if isinstance(exc, anthropic.NotFoundError):
        return f"Model not found: {exc.message}. Check llm.answer_model / llm.fast_model."
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude rate limit reached; retry shortly."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Claude API (network error)."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Claude API error {exc.status_code}: {exc.message}"
    return f"Claude API error: {exc}"
