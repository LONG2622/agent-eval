"""Tests for trace data models (TokenUsage, Span, RunRecord, AnnotationRecord)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_eval.trace.models import (
    AnnotationRecord,
    RunRecord,
    RunStatus,
    Span,
    SpanType,
    TokenUsage,
)

# ============================================================
# TokenUsage Tests
# ============================================================


class TestTokenUsage:
    def test_default_values(self):
        """TokenUsage should default to zero."""
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.total_tokens == 0

    def test_from_pair(self):
        """from_pair should compute total correctly."""
        t = TokenUsage.from_pair(100, 50)
        assert t.prompt_tokens == 100
        assert t.completion_tokens == 50
        assert t.total_tokens == 150

    def test_add(self):
        """add should sum tokens correctly."""
        t1 = TokenUsage.from_pair(100, 50)
        t2 = TokenUsage.from_pair(30, 20)
        result = t1.add(t2)
        assert result.prompt_tokens == 130
        assert result.completion_tokens == 70
        assert result.total_tokens == 200

    def test_add_zero(self):
        """Adding zero should preserve original."""
        t1 = TokenUsage.from_pair(100, 50)
        t2 = TokenUsage()
        result = t1.add(t2)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50


# ============================================================
# Span Tests
# ============================================================


class TestSpan:
    def test_create_span(self):
        """Span should be created with correct defaults."""
        span = Span(
            trace_id="run_001",
            span_type=SpanType.LLM_CALL,
            step_index=0,
            name="test_model",
        )
        assert span.trace_id == "run_001"
        assert span.span_type == SpanType.LLM_CALL
        assert span.step_index == 0
        assert span.name == "test_model"
        assert span.is_success is True
        assert span.span_id  # auto-generated
        assert span.created_at  # auto-generated timestamp

    def test_span_to_storage_dict(self):
        """to_storage_dict should produce JSON-serializable dict."""
        span = Span(
            trace_id="run_001",
            span_type=SpanType.TOOL_CALL,
            step_index=1,
            name="calculator",
            input_data={"arguments": {"a": 1}},
            output_data={"result": 1},
            latency_ms=100,
        )
        d = span.to_storage_dict()
        assert d["trace_id"] == "run_001"
        assert d["span_type"] == "tool_call"
        assert d["input_data"] == {"arguments": {"a": 1}}
        assert d["output_data"] == {"result": 1}
        assert isinstance(d["created_at"], str)

    def test_span_with_error(self):
        """Span with is_success=False should store exception."""
        span = Span(
            trace_id="run_001",
            span_type=SpanType.TOOL_CALL,
            is_success=False,
            exception="ValueError: invalid input",
        )
        assert span.is_success is False
        assert span.exception == "ValueError: invalid input"

    def test_span_types(self):
        """All SpanType values should exist."""
        assert SpanType.AGENT_STEP == "agent_step"
        assert SpanType.LLM_CALL == "llm_call"
        assert SpanType.TOOL_CALL == "tool_call"
        assert SpanType.THOUGHT == "thought"


# ============================================================
# RunRecord Tests
# ============================================================


class TestRunRecord:
    def test_create_run(self):
        """RunRecord should have correct defaults."""
        run = RunRecord(task_id="task_001", input_text="test task")
        assert run.status == RunStatus.PENDING
        assert run.run_id  # auto-generated
        assert run.trace_id == run.run_id
        assert run.started_at  # auto-generated

    def test_mark_finished_success(self):
        """mark_finished(SUCCESS) should update status and timestamp."""
        run = RunRecord()
        run.mark_finished(RunStatus.SUCCESS)
        assert run.status == RunStatus.SUCCESS
        assert run.finished_at is not None
        assert run.error_message is None

    def test_mark_finished_failed(self):
        """mark_finished(FAILED, error=...) should store error."""
        run = RunRecord()
        run.mark_finished(RunStatus.FAILED, error="LLM timeout")
        assert run.status == RunStatus.FAILED
        assert run.error_message == "LLM timeout"
        assert run.finished_at is not None

    def test_trace_id_property(self):
        """trace_id should equal run_id."""
        run = RunRecord(run_id="my_run_123")
        assert run.trace_id == "my_run_123"

    def test_to_storage_dict(self):
        """Storage dict should be JSON-serializable."""
        run = RunRecord(
            run_id="test_001",
            status=RunStatus.SUCCESS,
            final_output="42",
        )
        d = run.to_storage_dict()
        assert d["run_id"] == "test_001"
        assert d["status"] == "success"
        assert d["final_output"] == "42"


# ============================================================
# AnnotationRecord Tests
# ============================================================


class TestAnnotationRecord:
    def test_create_annotation(self):
        """AnnotationRecord should validate score range."""
        ann = AnnotationRecord(
            run_id="run_001",
            score=3,
            labels=["correct"],
        )
        assert ann.score == 3
        assert ann.annotator == "anonymous"  # default
        assert ann.labels == ["correct"]
        assert ann.annotation_id

    def test_score_validation_min(self):
        """Score < 1 should raise ValidationError."""
        with pytest.raises(ValidationError):
            AnnotationRecord(run_id="run_001", score=0)

    def test_score_validation_max(self):
        """Score > 5 should raise ValidationError."""
        with pytest.raises(ValidationError):
            AnnotationRecord(run_id="run_001", score=6)

    def test_score_boundary(self):
        """Score 1 and 5 should be valid."""
        ann1 = AnnotationRecord(run_id="run_001", score=1)
        ann5 = AnnotationRecord(run_id="run_001", score=5)
        assert ann1.score == 1
        assert ann5.score == 5

    def test_to_storage_dict(self):
        """Storage dict should be serializable."""
        ann = AnnotationRecord(run_id="run_001", score=4, labels=["correct", "fast"])
        d = ann.to_storage_dict()
        assert d["run_id"] == "run_001"
        assert d["score"] == 4
        assert d["labels"] == ["correct", "fast"]


# ============================================================
# RunStatus Tests
# ============================================================


class TestRunStatus:
    def test_all_statuses(self):
        """All expected status values should exist."""
        expected = {"pending", "running", "success", "failed", "timeout"}
        actual = {s.value for s in RunStatus}
        assert actual == expected
