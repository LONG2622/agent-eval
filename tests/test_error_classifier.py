"""Tests for the error classifier module."""

from __future__ import annotations

from agent_eval.evaluation.error_classifier import (
    ClassifiedError,
    ErrorCategory,
    ErrorSummary,
    classify_all_runs,
    classify_error,
    format_summary_text,
)
from agent_eval.trace.models import RunRecord, RunStatus

# ============================================================
# ErrorCategory Tests
# ============================================================


class TestErrorCategory:
    def test_category_matches(self):
        """matches() should detect error patterns (case-insensitive)."""
        cat = ErrorCategory(
            code="test",
            label="Test",
            description="Test category",
            patterns=["timeout", "timed out"],
        )
        assert cat.matches("Request timed out after 30s")
        assert cat.matches("TIMEOUT error occurred")
        assert not cat.matches("Everything fine")

    def test_category_no_match(self):
        """Empty patterns should never match."""
        cat = ErrorCategory(code="empty", label="Empty", description="", patterns=[])
        assert not cat.matches("anything")


# ============================================================
# classify_error Tests
# ============================================================


class TestClassifyError:
    def test_classify_timeout(self):
        """Should classify timeout errors."""
        cat = classify_error("APITimeoutError: request timed out")
        assert cat.code == "llm_timeout"

    def test_classify_rate_limit(self):
        """Should classify rate limit errors."""
        cat = classify_error("Rate limit exceeded: 429 Too Many Requests")
        assert cat.code == "llm_rate_limit"

    def test_classify_auth_error(self):
        """Should classify auth errors."""
        cat = classify_error("AuthenticationError: invalid API key (401)")
        assert cat.code == "llm_auth_error"

    def test_classify_network_error(self):
        """Should classify network errors."""
        cat = classify_error("ConnectionError: connection refused")
        assert cat.code == "network_error"

    def test_classify_internal_error(self):
        """Should classify internal code errors."""
        cat = classify_error("TypeError: 'NoneType' object has no attribute")
        assert cat.code == "internal_error"

    def test_classify_max_steps(self):
        """Should classify max steps exceeded."""
        cat = classify_error("Agent exceeded max_steps (10)")
        assert cat.code == "max_steps_exceeded"

    def test_classify_tool_error(self):
        """Should classify tool execution errors."""
        cat = classify_error("calculator tool execution failed")
        assert cat.code == "tool_execution_error"

    def test_classify_unknown(self):
        """Unrecognized errors should return UNKNOWN."""
        cat = classify_error("Something completely different happened")
        assert cat.code == "unknown"

    def test_classify_empty_error(self):
        """Empty error message should return UNKNOWN."""
        cat = classify_error("")
        assert cat.code == "unknown"

    def test_classify_with_span_data(self):
        """Should fall back to span data when error_message is empty."""
        span_data = [
            {"is_success": False, "output_data": {"error": "ConnectionError: connection refused"}},
        ]
        cat = classify_error("", span_data=span_data)
        assert cat.code == "network_error"

    def test_classify_span_data_tool_error(self):
        """Should detect tool errors from span data."""
        span_data = [
            {"is_success": False, "output_data": {"error": "calculator failed"}},
        ]
        cat = classify_error("", span_data=span_data)
        assert cat.code == "tool_execution_error"


# ============================================================
# ErrorSummary Tests
# ============================================================


class TestErrorSummary:
    def test_failure_rate(self):
        """failure_rate should be computed correctly."""
        summary = ErrorSummary(total_runs=10, total_failed=3, total_success=7)
        assert summary.failure_rate == 0.3

    def test_failure_rate_zero(self):
        """Zero runs should give 0.0 failure rate."""
        summary = ErrorSummary()
        assert summary.failure_rate == 0.0


# ============================================================
# classify_all_runs Tests
# ============================================================


class TestClassifyAllRuns:
    def test_classify_mixed_runs(self, storage):
        """Should classify a mix of success and failed runs."""
        # Add some success runs
        for i in range(3):
            run = RunRecord(
                run_id=f"success_{i}",
                status=RunStatus.SUCCESS,
                input_text=f"Task {i}",
                total_latency_ms=1000,
                total_steps=5,
            )
            storage.save_run(run)

        # Add some failed runs with different error types
        errors = [
            ("timeout_1", "APITimeoutError: request timed out"),
            ("rate_1", "Rate limit exceeded: 429"),
            ("auth_1", "AuthenticationError: invalid API key"),
        ]
        for run_id, error_msg in errors:
            run = RunRecord(
                run_id=run_id,
                status=RunStatus.FAILED,
                error_message=error_msg,
                input_text=f"Failed task {run_id}",
                total_latency_ms=500,
                total_steps=2,
            )
            storage.save_run(run)

        summary = classify_all_runs(storage)

        assert summary.total_runs == 6
        assert summary.total_success == 3
        assert summary.total_failed == 3
        assert summary.failure_rate == 0.5

        # Check categories
        assert "llm_timeout" in summary.by_category
        assert "llm_rate_limit" in summary.by_category
        assert "llm_auth_error" in summary.by_category

        # Check recent errors
        assert len(summary.recent_errors) == 3
        for err in summary.recent_errors:
            assert isinstance(err, ClassifiedError)
            assert err.category.code != "unknown"

    def test_classify_no_failed_runs(self, storage):
        """Only success runs should produce empty by_category."""
        for i in range(5):
            run = RunRecord(run_id=f"ok_{i}", status=RunStatus.SUCCESS)
            storage.save_run(run)

        summary = classify_all_runs(storage)
        assert summary.total_failed == 0
        assert len(summary.by_category) == 0
        assert len(summary.recent_errors) == 0

    def test_classify_with_limit(self, storage):
        """Limit should cap number of recent errors."""
        for i in range(10):
            run = RunRecord(
                run_id=f"fail_{i}",
                status=RunStatus.FAILED,
                error_message="timeout",
            )
            storage.save_run(run)

        summary = classify_all_runs(storage, limit=3)
        assert len(summary.recent_errors) == 3


# ============================================================
# format_summary_text Tests
# ============================================================


class TestFormatSummaryText:
    def test_format_with_data(self, storage):
        """format_summary_text should produce readable output."""
        run = RunRecord(
            run_id="fail_1",
            status=RunStatus.FAILED,
            error_message="timeout",
            input_text="Test task",
        )
        storage.save_run(run)
        summary = classify_all_runs(storage)
        text = format_summary_text(summary)
        assert "Total Runs" in text
        assert "Failure Rate" in text
        assert "Error Distribution" in text

    def test_format_empty(self):
        """Empty summary should still produce valid text."""
        summary = ErrorSummary()
        text = format_summary_text(summary)
        assert "Total Runs" in text
        assert "no failed runs" in text
