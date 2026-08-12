"""Top-level LLM Gateway - unified entry point for LLM calls."""

from __future__ import annotations

import logging
from typing import Any

from agent_eval.llm.messages import Message
from agent_eval.llm.providers.base import LLMCallback, LLMCallContext, LLMCallOptions, LLMProvider, LLMResponse
from agent_eval.llm.providers.openai_provider import OpenAIProvider
from agent_eval.llm.tokenizer import calculate_cost

logger = logging.getLogger("agent_eval.llm.gateway")


class LLMGateway:
    """Unified facade for LLM access with callbacks, retry, and metric hooks."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        callbacks: list[LLMCallback] | None = None,
    ) -> None:
        self._provider: LLMProvider = provider or OpenAIProvider()
        self._callbacks: list[LLMCallback] = list(callbacks or [])

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    def register_callback(self, cb: LLMCallback) -> None:
        self._callbacks.append(cb)

    def unregister_callback(self, cb: LLMCallback) -> None:
        if cb in self._callbacks:
            self._callbacks.remove(cb)

    @staticmethod
    def _normalize_messages(messages: list[Message | dict[str, Any]]) -> list[Message]:
        """Accept Message objects or raw dicts with 'role'+'content' keys."""
        normalized: list[Message] = []
        for m in messages:
            if isinstance(m, Message):
                normalized.append(m)
            elif isinstance(m, dict):
                normalized.append(Message(**m))
            else:
                raise TypeError(f"Unsupported message type: {type(m)}")
        return normalized

    def chat(
        self,
        messages: list[Message | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        timeout: int = 60,
    ) -> LLMResponse:
        messages = self._normalize_messages(messages)
        options = LLMCallOptions(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools or [],
            timeout=timeout,
        )
        ctx = LLMCallContext(
            model=options.model or "default",
            messages=messages,
            options=options,
        )
        for cb in self._callbacks:
            cb.on_call_start(ctx)

        try:
            response = self._provider.chat(messages, options)
            # Normalize cost if not attached by provider
            if not hasattr(response, "_cost"):
                response._cost = calculate_cost(  # type: ignore[attr-defined]
                    response.prompt_tokens,
                    response.completion_tokens,
                    response.model,
                )
            ctx.response = response
            for cb in self._callbacks:
                cb.on_call_end(ctx)
            return response
        except Exception as e:
            ctx.error = e
            for cb in self._callbacks:
                cb.on_call_error(ctx)
            raise
