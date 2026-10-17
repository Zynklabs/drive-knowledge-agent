"""Claude via the official ``anthropic`` SDK.

- Answers: streamed from the answer model (default Claude Opus 5.5) with explicit effort and
  server-side refusal fallback (``fallbacks="default"``).
- Small structured tasks (query rewrite, relevance grading): the fast model (default Claude
  Haiku 5.5) with structured outputs via ``messages.parse``. Haiku has no server-side fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from drive_agent.core.config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"
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


class ClaudeLLM:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self._config = settings.llm
        if client is None:
            key = settings.secrets.anthropic_api_key
            # No key: the SDK falls back to ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
            client = (
                anthropic.Anthropic(api_key=key.get_secret_value())
                if key
                else anthropic.Anthropic()
            )
        self._client = client

    def structured(self, system: str, prompt: str, schema: type[T]) -> T:
        """One fast-model call returning a validated ``schema`` instance."""
        try:
            response = self._client.messages.parse(
                model=self._config.fast_model,
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
            raise LLMError(
                f"{self._config.fast_model} returned no usable output ({response.stop_reason})"
            )
        return response.parsed_output

    def answer(
        self,
        system: str,
        messages: list[dict[str, Any]],
        on_text: Callable[[str], None] | None = None,
    ) -> Answer:
        """Stream an answer; ``on_text`` receives text deltas as they arrive."""
        try:
            with self._client.beta.messages.stream(
                model=self._config.answer_model,
                max_tokens=ANSWER_MAX_TOKENS,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,  # type: ignore[arg-type]
                output_config={"effort": self._config.answer_effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
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
