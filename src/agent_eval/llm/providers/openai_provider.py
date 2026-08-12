"""OpenAI API provider implementation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from agent_eval.config import load_config
from agent_eval.llm.messages import Message
from agent_eval.llm.providers.base import LLMCallOptions, LLMProvider, LLMResponse
from agent_eval.llm.tokenizer import calculate_cost, count_tokens_breakdown

logger = logging.getLogger("agent_eval.llm.openai")


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "openai package is required. Install with: pip install 'openai>=1.0'"
                ) from e
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: list[Message],
        options: LLMCallOptions | None = None,
    ) -> LLMResponse:
        cfg = load_config()
        options = options or LLMCallOptions()
        model = options.model or cfg.llm.default_model
        temperature = options.temperature if options.temperature is not None else cfg.llm.temperature
        max_tokens = options.max_tokens or cfg.llm.max_tokens

        dict_messages = [m.to_dict() for m in messages]

        retry_cfg = cfg.llm.retry
        last_error: Exception | None = None

        for attempt in range(1, retry_cfg.max_attempts + 1):
            try:
                return self._call_once(
                    model=model,
                    dict_messages=dict_messages,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    options=options,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt}/{retry_cfg.max_attempts} failed: {e}")
                if attempt < retry_cfg.max_attempts:
                    sleep_for = retry_cfg.backoff_factor ** (attempt - 1)
                    time.sleep(sleep_for)
        assert last_error is not None
        raise last_error

    def _call_once(
        self,
        *,
        model: str,
        dict_messages: list[dict[str, Any]],
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        options: LLMCallOptions,
    ) -> LLMResponse:
        client = self._get_client()
        started = time.perf_counter()

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": dict_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if options.tools:
            create_kwargs["tools"] = options.tools
            create_kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**create_kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)

        choice = resp.choices[0]
        message = choice.message
        content = message.content

        tool_calls: list[dict[str, Any]] | None = None
        if getattr(message, "tool_calls", None):
            import json

            tool_calls = []
            for tc in message.tool_calls:
                args_raw = tc.function.arguments
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": args_raw}
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args,
                    }
                )

        # Tokens: prefer provider's count, fall back to estimation
        if resp.usage is not None:
            prompt_tokens = resp.usage.prompt_tokens
            completion_tokens = resp.usage.completion_tokens
            total_tokens = resp.usage.total_tokens
        else:
            bd = count_tokens_breakdown(dict_messages, content or "", model=model)
            prompt_tokens = bd["prompt_tokens"]
            completion_tokens = bd["completion_tokens"]
            total_tokens = bd["total_tokens"]

        cost = calculate_cost(prompt_tokens, completion_tokens, model)

        response = LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            raw=resp,
        )
        # Store cost in extra (attached via gateway hooks)
        response._cost = cost  # type: ignore[attr-defined]
        return response
