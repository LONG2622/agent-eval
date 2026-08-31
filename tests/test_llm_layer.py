"""Tests for the LLM layer: tokenizer, messages, gateway, and OpenAI provider.

All LLM/network interactions are mocked - no real API calls are made.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_eval.config import PricingEntry
from agent_eval.llm.gateway import LLMGateway
from agent_eval.llm.messages import (
    Message,
    Role,
    ToolCall,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from agent_eval.llm.providers.base import (
    LLMCallback,
    LLMCallOptions,
    LLMProvider,
    LLMResponse,
)
from agent_eval.llm.providers.openai_provider import OpenAIProvider
from agent_eval.llm.tokenizer import (
    _get_encoder,
    calculate_cost,
    count_message_tokens,
    count_text_tokens,
)

# ============================================================
# Tokenizer
# ============================================================


class TestTokenizer:
    def test_count_tokens_english(self):
        text = "The quick brown fox jumps over the lazy dog near the river bank."
        n = count_text_tokens(text)
        assert n > 0
        assert n < len(text)  # tokens < chars for plain English

    def test_count_tokens_empty_string(self):
        assert count_text_tokens("") == 0

    def test_count_tokens_chinese(self):
        n = count_text_tokens("你好，世界！这是一个用于测试分词器的中文字符串。")
        assert n > 0

    def test_calculate_cost_with_known_pricing(self, monkeypatch):
        monkeypatch.setattr(
            "agent_eval.llm.tokenizer.get_pricing",
            lambda model: PricingEntry(prompt=0.5, completion=1.5),
        )
        cost = calculate_cost(1000, 500, "test-model")
        expected = (1000 / 1000) * 0.5 + (500 / 1000) * 1.5
        assert cost == pytest.approx(expected)

    def test_fallback_when_tiktoken_unavailable(self, monkeypatch):
        # Simulate tiktoken failing in both resolution paths:
        # encoding_for_model -> ImportError, get_encoding -> ValueError
        # so _get_encoder returns None and the char-based estimate kicks in.
        fake_tiktoken = MagicMock()
        fake_tiktoken.encoding_for_model.side_effect = ImportError("simulated: no model registry")
        fake_tiktoken.get_encoding.side_effect = ValueError("simulated: no encodings")
        monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

        _get_encoder.cache_clear()  # drop cached real encoders
        try:
            text = "This is a fallback token estimation test string"
            n = count_text_tokens(text, model="gpt-4o-mini")
            assert n == max(1, len(text) // 4)
            assert n > 0

            total = count_message_tokens(
                [{"role": "user", "content": "hello world"}], model="gpt-4o-mini"
            )
            # 4 per-message overhead + content estimate (11//4=2) + 2 priming
            assert total >= 4 + 2 + 2
        finally:
            _get_encoder.cache_clear()  # restore clean cache for other tests


# ============================================================
# Messages
# ============================================================


class TestMessages:
    def test_role_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_message_constructors(self):
        sys_msg = system_message("You are a helpful assistant.")
        assert sys_msg.role == Role.SYSTEM
        assert sys_msg.content == "You are a helpful assistant."

        user_msg = user_message("Hello!")
        assert user_msg.role == Role.USER
        assert user_msg.content == "Hello!"

        asst_msg = assistant_message("Hi there!")
        assert asst_msg.role == Role.ASSISTANT
        assert asst_msg.content == "Hi there!"

        tool_msg = tool_message("42.0", tool_call_id="call_1", name="calculator")
        assert tool_msg.role == Role.TOOL
        assert tool_msg.content == "42.0"
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.name == "calculator"

    def test_tool_call_model(self):
        tc = ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})
        assert tc.id == "call_1"
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "1+1"}

        default_tc = ToolCall(name="noop")
        assert default_tc.id is None
        assert default_tc.arguments == {}

    def test_to_dict_basic(self):
        assert user_message("hi").to_dict() == {"role": "user", "content": "hi"}
        assert system_message("sys prompt").to_dict() == {
            "role": "system",
            "content": "sys prompt",
        }

    def test_to_dict_tool_message(self):
        d = tool_message("result", tool_call_id="call_9", name="calculator").to_dict()
        assert d["role"] == "tool"
        assert d["content"] == "result"
        assert d["tool_call_id"] == "call_9"
        assert d["name"] == "calculator"

    def test_to_dict_assistant_with_tool_calls(self):
        msg = assistant_message(
            tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expression": "1+1"})]
        )
        d = msg.to_dict()
        assert d["role"] == "assistant"
        # Content is omitted when tool_calls are present (NVIDIA API requirement)
        assert "content" not in d
        assert len(d["tool_calls"]) == 1
        tc = d["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "calculator"
        assert json.loads(tc["function"]["arguments"]) == {"expression": "1+1"}

    def test_serialization_round_trip(self):
        for msg in (
            system_message("sys"),
            user_message("question?"),
            assistant_message("answer."),
            tool_message("tool result", tool_call_id="call_1", name="calc"),
        ):
            restored = Message(**msg.to_dict())
            assert restored.role == msg.role
            assert restored.content == msg.content
            assert restored.tool_call_id == msg.tool_call_id
            assert restored.name == msg.name


# ============================================================
# Gateway
# ============================================================


class FakeProvider(LLMProvider):
    """Provider stub returning a canned response, optionally failing N times first."""

    name = "fake"

    def __init__(self, response: LLMResponse | None = None, fail_times: int = 0) -> None:
        self.response = response or LLMResponse(
            content="fake answer",
            model="fake-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        self.fail_times = fail_times
        self.call_count = 0
        self.last_messages: list[Message] = []
        self.last_options: LLMCallOptions | None = None

    def chat(self, messages, options=None) -> LLMResponse:
        self.call_count += 1
        self.last_messages = list(messages)
        self.last_options = options
        if self.call_count <= self.fail_times:
            raise RuntimeError("simulated provider failure")
        return self.response


class RecordingCallback(LLMCallback):
    """Callback that records every hook invocation."""

    def __init__(self) -> None:
        self.starts: list = []
        self.ends: list = []
        self.errors: list = []

    def on_call_start(self, ctx) -> None:
        self.starts.append(ctx)

    def on_call_end(self, ctx) -> None:
        self.ends.append(ctx)

    def on_call_error(self, ctx) -> None:
        self.errors.append(ctx)


class TestGateway:
    def test_chat_returns_response(self, patch_config):
        gw = LLMGateway(provider=FakeProvider())
        resp = gw.chat([{"role": "user", "content": "hi"}])
        assert resp.content == "fake answer"
        assert resp.model == "fake-model"
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50
        assert resp.total_tokens == 150
        # Raw dicts are normalized to Message objects before hitting the provider
        assert isinstance(gw.provider.last_messages[0], Message)
        assert gw.provider.last_messages[0].content == "hi"

    def test_gateway_normalizes_cost(self, patch_config):
        cfg = patch_config
        cfg.pricing["fake-model"] = PricingEntry(prompt=1.0, completion=2.0)
        gw = LLMGateway(provider=FakeProvider())
        resp = gw.chat([user_message("hi")])
        expected = (100 / 1000) * 1.0 + (50 / 1000) * 2.0
        assert resp._cost == pytest.approx(expected)

    def test_callback_on_response_fired(self, patch_config):
        gw = LLMGateway(provider=FakeProvider())
        cb = RecordingCallback()
        gw.register_callback(cb)

        resp = gw.chat([user_message("hi")], model="fake-model")

        assert len(cb.starts) == 1
        assert len(cb.ends) == 1
        assert cb.ends[0].response is resp
        assert cb.starts[0].model == "fake-model"
        assert cb.errors == []

    def test_provider_error_raises_and_error_callback_fired(self, patch_config):
        gw = LLMGateway(provider=FakeProvider(fail_times=99))
        cb = RecordingCallback()
        gw.register_callback(cb)

        with pytest.raises(RuntimeError, match="simulated provider failure"):
            gw.chat([user_message("hi")])

        assert len(cb.errors) == 1
        assert isinstance(cb.errors[0].error, RuntimeError)
        assert cb.ends == []
        # The gateway itself performs a single attempt (no internal retry loop)
        assert gw.provider.call_count == 1


# ============================================================
# OpenAI Provider (mocked openai client)
# ============================================================


def _make_completion(content="Hello!", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    """Build a fake OpenAI chat completion response object."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self, patch_config):
        cfg = patch_config
        cfg.llm.retry.max_attempts = 2
        cfg.llm.retry.backoff_factor = 0.0
        return OpenAIProvider(api_key="test-key-123456", base_url="https://fake.example/v1")

    def _patch_client(self, monkeypatch, client) -> None:
        # openai_provider does `from openai import OpenAI` at call time,
        # so patching the attribute on the real module is picked up.
        monkeypatch.setattr("openai.OpenAI", MagicMock(return_value=client))

    def test_basic_chat(self, provider, monkeypatch):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_completion()
        self._patch_client(monkeypatch, client)

        messages = [system_message("You are terse."), user_message("hi")]
        options = LLMCallOptions(model="gpt-4o-mini", temperature=0.2, max_tokens=100)
        resp = provider.chat(messages, options)

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o-mini"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.total_tokens == 15
        assert resp.tool_calls is None
        assert resp.latency_ms >= 0
        assert hasattr(resp, "_cost")  # cost attached by the provider

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 100
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ]

    def test_response_with_tool_calls_parsed(self, provider, monkeypatch):
        client = MagicMock()
        fake_tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="calculator", arguments='{"expression": "sqrt(144)"}'),
        )
        client.chat.completions.create.return_value = _make_completion(
            content=None, tool_calls=[fake_tc]
        )
        self._patch_client(monkeypatch, client)

        resp = provider.chat(
            [user_message("What is the square root of 144?")],
            LLMCallOptions(model="gpt-4o-mini"),
        )

        assert resp.has_tool_calls is True
        assert resp.content is None
        assert resp.tool_calls == [
            {"id": "call_1", "name": "calculator", "arguments": {"expression": "sqrt(144)"}}
        ]

    def test_client_connection_error_retries_then_raises(self, provider, monkeypatch, patch_config):
        cfg = patch_config
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("connection refused")
        self._patch_client(monkeypatch, client)
        # Skip retry backoff sleeps
        monkeypatch.setattr("agent_eval.llm.providers.openai_provider.time.sleep", lambda _s: None)

        with pytest.raises(ConnectionError, match="connection refused"):
            provider.chat([user_message("hi")], LLMCallOptions(model="gpt-4o-mini"))

        assert client.chat.completions.create.call_count == cfg.llm.retry.max_attempts == 2
