"""LLM Provider abstract base class and registry."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_eval.llm.messages import Message


@dataclass
class LLMResponse:
    """Standard response wrapper for all LLM providers."""

    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    raw: Any = None  # Original provider response

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class LLMCallOptions:
    """Per-request options that override defaults."""

    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    timeout: int = 60


class LLMProvider(ABC):
    """Abstract base for a model provider (OpenAI, Anthropic, Ollama, etc.)."""

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        options: LLMCallOptions | None = None,
    ) -> LLMResponse:
        """Run a chat completion and return a standardized LLMResponse."""
        raise NotImplementedError


# -------------------- Callback Hook --------------------


@dataclass
class LLMCallContext:
    """Context passed to LLM callbacks (before/after a call)."""

    model: str
    messages: list[Message]
    options: LLMCallOptions
    started_at: float = field(default_factory=time.time)
    response: LLMResponse | None = None
    error: Exception | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMCallback(ABC):
    """Hook for tracing/observing LLM calls."""

    def on_call_start(self, ctx: LLMCallContext) -> None:
        pass

    def on_call_end(self, ctx: LLMCallContext) -> None:
        pass

    def on_call_error(self, ctx: LLMCallContext) -> None:
        pass
