"""Tests for TraceRecorder: lifecycle, callback hooks, and JSONL persistence.

Pure-logic tests - no LLM and no network involved.
"""

from __future__ import annotations

import pytest

from agent_eval.llm import user_message
from agent_eval.llm.providers.base import LLMCallContext, LLMCallOptions, LLMResponse
from agent_eval.tools import ToolCallContext, ToolResult
from agent_eval.trace import (
    RunRecord,
    RunStatus,
    SpanType,
    TraceRecorder,
)


def _make_run(task_id: str = "task_rec") -> RunRecord:
    return RunRecord(
        task_id=task_id,
        agent_name="recorder-test-agent",
        input_text="hello recorder",
    )


def _make_llm_ctx(
    content: str = "hi there",
    tool_calls: list[dict] | None = None,
    model: str = "fake-model",
) -> LLMCallContext:
    resp = LLMResponse(
        content=content,
        tool_calls=tool_calls,
        model=model,
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=123,
    )
    resp._cost = 0.002  # normally attached by LLMGateway
    ctx = LLMCallContext(
        model=model,
        messages=[user_message("hello")],
        options=LLMCallOptions(model=model),
    )
    ctx.response = resp
    return ctx


# -------------------- Run lifecycle --------------------


def test_start_run_persists_running_record(storage, tmp_output_dir):
    recorder = TraceRecorder(storage)
    run = _make_run()

    returned = recorder.start_run(run)

    assert returned is run
    assert run.status == RunStatus.RUNNING

    loaded = storage.load_run(run.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.RUNNING
    assert loaded.input_text == "hello recorder"
    assert loaded.task_id == "task_rec"
    assert (tmp_output_dir / "runs" / "runs.jsonl").exists()


def test_end_run_persists_status_output_and_step_spans(storage, tmp_output_dir):
    recorder = TraceRecorder(storage)
    run = _make_run()
    recorder.start_run(run)

    recorder.on_step_start(run, 1, "thinking about it")
    recorder.on_step_end(run, 1, "calculator(2+3)", "[calculator] 5")

    finished = recorder.end_run(run, status=RunStatus.SUCCESS, final_output="5")

    assert finished is run
    assert run.status == RunStatus.SUCCESS
    assert run.final_output == "5"
    assert run.finished_at is not None
    assert run.error_message is None

    spans = storage.load_spans(run.run_id)
    assert len(spans) == 2
    assert all(s.span_type == SpanType.AGENT_STEP for s in spans)
    assert all(s.trace_id == run.run_id for s in spans)

    start_spans = [s for s in spans if s.metadata.get("phase") == "start"]
    end_spans = [s for s in spans if s.metadata.get("phase") == "end"]
    assert len(start_spans) == 1 and len(end_spans) == 1
    assert start_spans[0].input_data == {"thought": "thinking about it"}
    assert start_spans[0].name == "step_1"
    assert end_spans[0].output_data == {
        "action": "calculator(2+3)",
        "observation": "[calculator] 5",
    }
    assert end_spans[0].name == "step_1_end"

    # Spans are flushed into the per-trace JSONL file under trace_dir.
    trace_file = tmp_output_dir / "traces" / f"{run.run_id}.jsonl"
    assert trace_file.exists()
    assert len(trace_file.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_end_run_failure_sets_error_message(storage):
    recorder = TraceRecorder(storage)
    run = _make_run()
    recorder.start_run(run)
    recorder.on_step_start(run, 1, "will explode")

    finished = recorder.end_run(run, status=RunStatus.FAILED, error="ValueError: bad input")

    assert finished.status == RunStatus.FAILED
    assert finished.error_message == "ValueError: bad input"
    assert finished.finished_at is not None
    assert storage.load_run(run.run_id).status == RunStatus.FAILED


def test_multiple_runs_tracked_in_storage(storage):
    recorder = TraceRecorder(storage)
    run_ids = []
    for i, task_id in enumerate(["r1", "r2"], start=1):
        run = _make_run(task_id)
        run_ids.append(run.run_id)
        recorder.start_run(run)
        recorder.on_step_start(run, i, f"thought {i}")
        recorder.end_run(run, status=RunStatus.SUCCESS, final_output=f"out {i}")

    runs = storage.list_runs()
    assert {r.run_id for r in runs} == set(run_ids)
    assert all(r.status == RunStatus.SUCCESS for r in runs)
    for run_id in run_ids:
        assert len(storage.load_spans(run_id)) == 1


# -------------------- LLM callback hooks --------------------


def test_llm_callback_records_llm_span_and_aggregates(storage):
    recorder = TraceRecorder(storage)
    run = _make_run()
    recorder.start_run(run)

    recorder.on_call_end(_make_llm_ctx(content="hello answer"))

    finished = recorder.end_run(run, status=RunStatus.SUCCESS, final_output="hello answer")

    spans = storage.load_spans(run.run_id)
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.LLM_CALL
    assert span.name == "fake-model"
    assert span.output_data["content"] == "hello answer"
    assert span.tokens.total_tokens == 15
    assert span.latency_ms == 123
    assert span.is_success is True

    # Run-level aggregation from spans
    assert finished.tokens.total_tokens == 15
    assert finished.total_cost == pytest.approx(0.002)


def test_llm_callback_records_tool_calls_in_span(storage):
    recorder = TraceRecorder(storage)
    run = _make_run()
    recorder.start_run(run)

    tool_calls = [{"id": "call_1", "name": "calculator", "arguments": {"expression": "2+3"}}]
    recorder.on_call_end(_make_llm_ctx(content="using a tool", tool_calls=tool_calls))
    recorder.end_run(run, status=RunStatus.SUCCESS, final_output="5")

    spans = storage.load_spans(run.run_id)
    assert len(spans) == 1
    assert spans[0].output_data["tool_calls"] == tool_calls


def test_llm_error_callback_records_failed_span(storage):
    recorder = TraceRecorder(storage)
    run = _make_run()
    recorder.start_run(run)

    ctx = _make_llm_ctx()
    ctx.response = None
    ctx.error = RuntimeError("boom")
    recorder.on_call_error(ctx)

    finished = recorder.end_run(run, status=RunStatus.FAILED, error="RuntimeError: boom")

    spans = storage.load_spans(run.run_id)
    assert len(spans) == 1
    span = spans[0]
    assert span.span_type == SpanType.LLM_CALL
    assert span.is_success is False
    assert span.exception == "RuntimeError: boom"
    assert "messages" in span.input_data
    assert finished.status == RunStatus.FAILED
    assert finished.error_message == "RuntimeError: boom"


# -------------------- Tool callback hooks --------------------


def test_tool_callback_records_success_and_error_spans(storage):
    recorder = TraceRecorder(storage)
    run = _make_run()
    recorder.start_run(run)

    ok_ctx = ToolCallContext(tool_name="calculator", arguments={"expression": "2+3"})
    ok_ctx.result = ToolResult(output="5", success=True, latency_ms=42)
    recorder.on_tool_end(ok_ctx)

    bad_ctx = ToolCallContext(tool_name="web_search", arguments={"query": "x"})
    bad_ctx.error = ValueError("bad query")
    recorder.on_tool_error(bad_ctx)

    recorder.end_run(run, status=RunStatus.SUCCESS, final_output="5")

    tool_spans = [s for s in storage.load_spans(run.run_id) if s.span_type == SpanType.TOOL_CALL]
    assert len(tool_spans) == 2

    ok = next(s for s in tool_spans if s.name == "calculator")
    assert ok.is_success is True
    assert ok.input_data["arguments"] == {"expression": "2+3"}
    assert ok.output_data["output"] == "5"
    assert ok.output_data["success"] is True
    assert ok.latency_ms == 42
    assert ok.exception is None

    bad = next(s for s in tool_spans if s.name == "web_search")
    assert bad.is_success is False
    assert bad.exception == "ValueError: bad query"


# -------------------- Buffering / no-active-run behaviour --------------------


def test_spans_only_persisted_after_end_run(storage):
    recorder = TraceRecorder(storage)
    run = _make_run()
    recorder.start_run(run)

    recorder.on_call_end(_make_llm_ctx())
    # Still buffered in memory - nothing written yet.
    assert storage.load_spans(run.run_id) == []

    recorder.end_run(run, status=RunStatus.SUCCESS, final_output="x")
    assert len(storage.load_spans(run.run_id)) == 1


def test_callbacks_without_active_run_are_dropped(storage):
    recorder = TraceRecorder(storage)
    ctx = _make_llm_ctx()

    # 1) Before any run started: callbacks are safe no-ops.
    recorder.on_call_end(ctx)
    recorder.on_call_error(ctx)

    # 2) After a run has ended (active run cleared): also dropped.
    run1 = _make_run("task_first")
    recorder.start_run(run1)
    recorder.end_run(run1, status=RunStatus.SUCCESS, final_output="done")
    recorder.on_call_end(ctx)
    recorder.on_tool_end(ToolCallContext(tool_name="calculator", arguments={}))

    # A brand-new run must not contain any phantom spans.
    run2 = _make_run("task_second")
    recorder.start_run(run2)
    recorder.end_run(run2, status=RunStatus.SUCCESS, final_output="done")
    assert storage.load_spans(run2.run_id) == []
