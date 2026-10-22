"""Groq chat client with retries, model fallback, timing and token usage.

The API key is read from the GROQ_API_KEY environment variable (.env), never from code.

Rate limits (HTTP 429):
- per-minute limits: the SDK waits and retries automatically (max_retries)
- per-day limits (TPD/RPD) are counted per model: the client switches to the next model of
  `fallback_models` (the app stays usable). Evaluations disable the fallback, so that every
  answer of a run comes from the same model, and stop cleanly instead (DailyLimitError).
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


class DailyLimitError(RuntimeError):
    """Every allowed model reached its daily quota."""


def is_daily_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text


@dataclass
class LLMResponse:
    text: str
    model: str
    seconds: float
    usage: dict[str, int] = field(default_factory=dict)


class GroqLLM:
    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 10,
                 api_key: str | None = None, fallback_models: list[str] | None = None) -> None:
        from groq import Groq

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key or key.startswith("your_"):
            raise RuntimeError("GROQ_API_KEY is missing: put your key in the .env file")
        self.client = Groq(api_key=key, max_retries=max_retries)
        self.models = [model, *(fallback_models or [])]
        self.temperature = temperature

    @property
    def model(self) -> str:
        """The model currently in use (changes after a daily limit, if fallback is on)."""
        return self.models[0]

    @classmethod
    def from_settings(cls, settings: dict[str, Any], role: str = "model",
                      fallback: bool = True) -> GroqLLM:
        cfg = settings["llm"]
        fallbacks = cfg.get("fallback_models", []) if fallback and role == "model" else []
        return cls(cfg[role], cfg.get("temperature", 0.0), cfg.get("max_retries", 10),
                   fallback_models=fallbacks)

    def _call(self, request: Callable[[str], T]) -> T:
        """Run request(model); on a daily limit, move on to the next model."""
        from groq import RateLimitError

        while True:
            try:
                return request(self.model)
            except RateLimitError as error:
                if not is_daily_limit(error):
                    raise
                if len(self.models) == 1:
                    raise DailyLimitError(f"Daily Groq limit reached for {self.model}. {error}") from error
                self.models.pop(0)  # this model is done for today: use the next one

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn:
        return complete(self, messages, tools)

    def chat(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        from groq import BadRequestError

        start = time.perf_counter()
        kwargs: dict[str, Any] = {"response_format": {"type": "json_object"}} if json_mode else {}

        def request(model: str, **extra: Any) -> Any:
            return self.client.chat.completions.create(
                model=model, temperature=self.temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)

        try:
            response = self._call(lambda model: request(model, **kwargs))
        except BadRequestError as error:
            # JSON mode: Groq rejects the call when the model's output is not valid JSON
            # (e.g. empty). Retry once in normal mode; the caller parses the text itself.
            if not json_mode or "json_validate_failed" not in str(error):
                raise
            response = self._call(lambda model: request(model))
        usage = response.usage
        return LLMResponse(
            text=(response.choices[0].message.content or "").strip(),
            model=response.model or self.model,
            seconds=time.perf_counter() - start,
            usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
            if usage else {})


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as produced by the model


@dataclass
class ChatTurn:
    """One model turn in a tool-calling conversation."""
    content: str
    tool_calls: list[ToolCall]
    seconds: float
    usage: dict[str, int] = field(default_factory=dict)

    def as_message(self) -> dict[str, Any]:
        """The assistant message to append to the conversation history."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": c.arguments}}
                                     for c in self.tool_calls]
        return message


def complete(llm: GroqLLM, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ChatTurn:
    """One call with tools available; the model either calls tools or answers.
    Used through GroqLLM.complete."""
    start = time.perf_counter()
    response = llm._call(lambda model: llm.client.chat.completions.create(
        model=model, temperature=llm.temperature, messages=messages, tools=tools, tool_choice="auto"))
    message = response.choices[0].message
    usage = response.usage
    return ChatTurn(
        content=(message.content or "").strip(),
        tool_calls=[ToolCall(c.id, c.function.name, c.function.arguments or "{}")
                    for c in (message.tool_calls or [])],
        seconds=time.perf_counter() - start,
        usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
        if usage else {})
