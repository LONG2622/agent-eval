"""Tests for ReActAgent with a fully scripted (mocked) LLM provider.

Zero real network calls: LLMGateway is constructed with a fake provider whose
chat() returns pre-scripted LLMResponses (or raises).
"""

from __future__ import annotations

import pytest

from agent_eval.agent import AgentRunConfig
from agent_eval.agent.react_agent import ReActAgent
from agent_eval.config import LLMModelProfile
from agent_eval.llm import LLMGateway, LLMResponse, Role
from agent_eval.llm.providers.base import LLMCallOptions, LLMProvider
from agent_eval.tools import ToolRegistry, register_builtin_tools
from agent_eval.trace import JSONLStorage, RunStatus, SpanType, TraceRecorder

# -------------------- Fake LLM infrastructure --------------------


class ScriptedProvider(LLMProvider):
    """Provider that replays scripted LLMResponse objects (or raises)."""

    name = "scripted"

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []
        self.options_seen: list[LLMCallOptions | None] = []

    def chat(self, messages, options=None) -> LLMResponse:
        self.calls.append(list(messages))
        self.options_seen.append(options)
        if not self._responses:
            raise RuntimeError("Scripted provider ran out of responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _response(content: str | None = None, tool_calls=None, model: str = "fake-model") -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        model=model,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=7,
    )


# ReAct-formatted scratchpad text that triggers a calculator tool call.
SCRATCHPAD_TOOL_CALL = (
    "Thought: I need to compute 2+3.\n"
    "Action: calculator\n"
    'Action Input: {"expression": "2+3"}'
)


@pytest.fixture
def agent_env(tmp_path):
    """Factory building a ReActAgent wired to a ScriptedProvider and tmp storage."""

    def _build(responses: list, *, model: str | None = None, max_steps: int = 5):
        storage = JSONLStorage(
            trace_dir=tmp_path / "traces",
            run_dir=tmp_path / "runs",
            annotation_dir=tmp_path / "annotations",
        )
        recorder = TraceRecorder(storage)
        registry = ToolRegistry()
        register_builtin_tools(registry)  # real builtin calculator tool
        provider = ScriptedProvider(responses)
        gateway = LLMGateway(provider=provider)
        config = AgentRunConfig(
            agent_name="test-react",
            model=model,
            max_steps=max_steps,
        )
        agent = ReActAgent(gateway, registry, recorder, config)
        return agent, provider, storage

    return _build


@pytest.fixture
def scratchpad_mode(monkeypatch):
    """Force the scratchpad fallback path (model profile without function calling)."""
    profile = LLMModelProfile(
        id="fake-model",
        display_name="Fake Model",
        model="fake-model",
        supports_function_calling=False,
    )
    monkeypatch.setattr("agent_eval.config.get_model_profile", lambda model=None: profile)


# -------------------- Scratchpad (fallback) mode --------------------


def test_scratchpad_tool_call_then_final_answer(agent_env, scratchpad_mode):
    agent, provider, storage = agent_env(
        [
            _response(content=SCRATCHPAD_TOOL_CALL),
            _response(
                content=(
                    "Thought: Done.\n"
                    "Action: Final Answer\n"
                    "Action Input: The answer is 5."
                )
            ),
        ],
        model="fake-model",
    )

    output, run = agent.run("What is 2+3?")

    assert run.status == RunStatus.SUCCESS
    assert output == "The answer is 5."
    assert run.final_output == "The answer is 5."
    assert run.error_message is None
    assert run.finished_at is not None

    spans = storage.load_spans(run.run_id)
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM_CALL]
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL_CALL]
    assert len(llm_spans) == 2
    assert len(tool_spans) == 1

    # The real builtin calculator was invoked and returned 5 for "2+3".
    tool_span = tool_spans[0]
    assert tool_span.name == "calculator"
    assert tool_span.input_data["arguments"] == {"expression": "2+3"}
    assert tool_span.output_data["output"] == "5"
    assert tool_span.is_success is True
    # The Observation fed back to the model contains the tool result.
    observations = [s.output_data.get("observation", "") for s in spans]
    assert "[calculator] 5" in observations

    # Scratchpad mode must NOT send function schemas to the LLM.
    assert provider.options_seen[0].tools == []

    # Token usage aggregated from both LLM calls (10 + 5 each).
    assert run.tokens.total_tokens == 30


def test_scratchpad_direct_final_answer_without_tools(agent_env, scratchpad_mode):
    agent, provider, storage = agent_env(
        [
            _response(
                content=(
                    "Thought: I already know this.\n"
                    "Action: Final Answer\n"
                    "Action Input: Paris is the capital of France."
                )
            )
        ],
        model="fake-model",
        max_steps=3,
    )

    output, run = agent.run("What is the capital of France?")

    assert run.status == RunStatus.SUCCESS
    assert output == "Paris is the capital of France."
    assert run.final_output == "Paris is the capital of France."

    spans = storage.load_spans(run.run_id)
    assert len([s for s in spans if s.span_type == SpanType.LLM_CALL]) == 1
    assert not [s for s in spans if s.span_type == SpanType.TOOL_CALL]


def test_plain_content_treated_as_final_answer(agent_env, scratchpad_mode):
    """Content without any Thought/Action markers ends the run directly."""
    agent, provider, storage = agent_env(
        [_response(content="Paris is the capital of France.")],
        model="fake-model",
    )

    output, run = agent.run("What is the capital of France?")

    assert run.status == RunStatus.SUCCESS
    assert output == "Paris is the capital of France."
    spans = storage.load_spans(run.run_id)
    assert len([s for s in spans if s.span_type == SpanType.LLM_CALL]) == 1
    assert not [s for s in spans if s.span_type == SpanType.TOOL_CALL]


def test_max_steps_exhaustion_ends_with_placeholder(agent_env, scratchpad_mode):
    """LLM keeps requesting tools and never produces a final answer.

    Correct semantics: run ends FAILED with error_message set, because the
    agent did not actually complete the task.
    """
    agent, provider, storage = agent_env(
        [_response(content=SCRATCHPAD_TOOL_CALL) for _ in range(2)],
        model="fake-model",
        max_steps=2,
    )

    output, run = agent.run("What is 2+3?")

    assert run.status == RunStatus.FAILED
    assert run.error_message and "max steps" in run.error_message.lower()
    assert "exceeded max steps" in output.lower()

    spans = storage.load_spans(run.run_id)
    assert len([s for s in spans if s.span_type == SpanType.LLM_CALL]) == 2
    assert len([s for s in spans if s.span_type == SpanType.TOOL_CALL]) == 2


def test_unknown_tool_fails_run(agent_env, scratchpad_mode):
    bad_call = (
        "Thought: I need a mystery tool.\n"
        "Action: does_not_exist\n"
        'Action Input: {"x": 1}'
    )
    agent, provider, storage = agent_env(
        [_response(content=bad_call)],
        model="fake-model",
    )

    output, run = agent.run("Use the mystery tool")

    assert run.status == RunStatus.FAILED
    assert "KeyError" in run.error_message
    assert run.error_message.startswith("KeyError")
    assert output.startswith("Error:")


# -------------------- Error propagation --------------------


def test_llm_error_propagates_to_failed_run(agent_env, scratchpad_mode):
    agent, provider, storage = agent_env(
        [RuntimeError("LLM provider exploded")],
        model="fake-model",
    )

    output, run = agent.run("What is 2+3?")

    assert run.status == RunStatus.FAILED
    assert run.error_message == "RuntimeError: LLM provider exploded"
    assert output == "Error: LLM provider exploded"
    assert run.finished_at is not None

    # The gateway error hook recorded a failed LLM_CALL span.
    llm_spans = [s for s in storage.load_spans(run.run_id) if s.span_type == SpanType.LLM_CALL]
    assert len(llm_spans) == 1
    assert llm_spans[0].is_success is False
    assert "RuntimeError" in llm_spans[0].exception


# -------------------- Function-calling mode --------------------


def test_function_calling_mode_invokes_tool(agent_env):
    """Default path (no model profile): tools registered -> function calling."""
    agent, provider, storage = agent_env(
        [
            _response(
                content="I'll use the calculator.",
                tool_calls=[
                    {"id": "call_1", "name": "calculator", "arguments": {"expression": "2+3"}}
                ],
                model="fake-fc-model",
            ),
            _response(content="The result is 5.", model="fake-fc-model"),
        ],
        model=None,  # no profile lookup -> FC enabled because tools exist
    )

    output, run = agent.run("What is 2+3?")

    assert run.status == RunStatus.SUCCESS
    assert output == "The result is 5."

    # Function schemas were sent on the first call.
    assert provider.options_seen[0].tools
    assert provider.options_seen[0].tools[0]["function"]["name"] == "calculator"

    spans = storage.load_spans(run.run_id)
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL_CALL]
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "calculator"
    assert tool_spans[0].output_data["output"] == "5"

    # The second LLM call saw the assistant tool-call message and the tool result.
    second_call = provider.calls[1]
    assert any(m.role == Role.ASSISTANT and m.tool_calls for m in second_call)
    assert any(m.role == Role.TOOL and m.content == "5" for m in second_call)
