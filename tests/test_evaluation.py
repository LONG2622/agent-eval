"""Tests for the 5 built-in evaluators."""

from __future__ import annotations

import pytest

from agent_eval.evaluation.builtin import (
    AnswerQualityEvaluator,
    LatencyEvaluator,
    SuccessRateEvaluator,
    TokenCostEvaluator,
    ToolUsageEvaluator,
)
from agent_eval.evaluation.base import EvalDimension, SubMetric
from agent_eval.trace.models import RunRecord, RunStatus, Span, SpanType, TokenUsage


# ============================================================
# SuccessRateEvaluator Tests
# ============================================================


class TestSuccessRateEvaluator:
    def test_success_status(self, sample_run, sample_spans):
        """SUCCESS run should get pass score 1.0."""
        evaluator = SuccessRateEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        pass_result = [r for r in results if r.sub_metric == SubMetric.PASS][0]
        assert pass_result.score == 1.0
        assert pass_result.passed is True

    def test_failed_status(self, sample_spans):
        """FAILED run should get pass score 0.0."""
        run = RunRecord(status=RunStatus.FAILED, error_message="error")
        evaluator = SuccessRateEvaluator()
        results = evaluator.evaluate(run, sample_spans)
        pass_result = [r for r in results if r.sub_metric == SubMetric.PASS][0]
        assert pass_result.score == 0.0
        assert pass_result.passed is False

    def test_keyword_match(self, sample_spans):
        """Should evaluate keyword match when expected_output is set."""
        run = RunRecord(
            status=RunStatus.SUCCESS,
            final_output="The answer is 37",
            expected_output="37",
        )
        evaluator = SuccessRateEvaluator()
        results = evaluator.evaluate(run, sample_spans)
        keyword_results = [r for r in results if r.sub_metric == SubMetric.KEYWORD_MATCH]
        assert len(keyword_results) == 1
        assert keyword_results[0].score >= 0.0

    def test_no_expected_output(self, sample_run, sample_spans):
        """Without expected_output, no keyword match result."""
        run = RunRecord(status=RunStatus.SUCCESS)
        evaluator = SuccessRateEvaluator()
        results = evaluator.evaluate(run, sample_spans)
        keyword_results = [r for r in results if r.sub_metric == SubMetric.KEYWORD_MATCH]
        assert len(keyword_results) == 0


# ============================================================
# ToolUsageEvaluator Tests
# ============================================================


class TestToolUsageEvaluator:
    def test_tool_call_count(self, sample_run, sample_spans):
        """Should count tool calls correctly."""
        evaluator = ToolUsageEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        count_result = [r for r in results if r.sub_metric == SubMetric.TOOL_CALL_COUNT][0]
        assert count_result.score == 1  # one tool_call span

    def test_tool_success_rate(self, sample_run, sample_spans):
        """All successful tools should give 100% success rate."""
        evaluator = ToolUsageEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        rate_result = [r for r in results if r.sub_metric == SubMetric.TOOL_SUCCESS_RATE][0]
        assert rate_result.score == 1.0
        assert rate_result.passed is True

    def test_redundant_calls_zero(self, sample_run, sample_spans):
        """No redundant calls should be detected."""
        evaluator = ToolUsageEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        redundant_result = [r for r in results if r.sub_metric == SubMetric.REDUNDANT_CALLS][0]
        assert redundant_result.score == 0
        assert redundant_result.passed is True

    def test_redundant_calls_detected(self, sample_run):
        """Duplicate tool calls should be detected as redundant."""
        spans = [
            Span(trace_id="test", span_type=SpanType.TOOL_CALL, name="calc",
                 input_data={"arguments": {"a": 1}}, is_success=True),
            Span(trace_id="test", span_type=SpanType.TOOL_CALL, name="calc",
                 input_data={"arguments": {"a": 1}}, is_success=True),  # duplicate
        ]
        evaluator = ToolUsageEvaluator()
        results = evaluator.evaluate(sample_run, spans)
        redundant_result = [r for r in results if r.sub_metric == SubMetric.REDUNDANT_CALLS][0]
        assert redundant_result.score == 1
        assert redundant_result.passed is False

    def test_no_tool_calls(self, sample_run):
        """Empty tool calls list should have success rate 1.0."""
        spans = [Span(trace_id="test", span_type=SpanType.AGENT_STEP)]
        evaluator = ToolUsageEvaluator()
        results = evaluator.evaluate(sample_run, spans)
        rate_result = [r for r in results if r.sub_metric == SubMetric.TOOL_SUCCESS_RATE][0]
        assert rate_result.score == 1.0


# ============================================================
# AnswerQualityEvaluator Tests
# ============================================================


class TestAnswerQualityEvaluator:
    def test_quality_good_answer(self, sample_run, sample_spans):
        """Good answer should pass quality checks."""
        evaluator = AnswerQualityEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        overall = [r for r in results if r.sub_metric is None][0]
        assert overall.score >= 0.6
        assert overall.passed is True

    def test_quality_empty_output(self, sample_spans):
        """Empty output should fail completeness."""
        run = RunRecord(status=RunStatus.SUCCESS, final_output="")
        evaluator = AnswerQualityEvaluator()
        results = evaluator.evaluate(run, sample_spans)
        completeness = [r for r in results if r.sub_metric == SubMetric.COMPLETENESS][0]
        assert completeness.score == 0.0
        assert completeness.passed is False

    def test_quality_refusal_detected(self, sample_spans):
        """Refusal patterns should lower relevance."""
        run = RunRecord(
            status=RunStatus.SUCCESS,
            final_output="I cannot answer this question.",
            expected_output="42",
        )
        evaluator = AnswerQualityEvaluator()
        results = evaluator.evaluate(run, sample_spans)
        relevance = [r for r in results if r.sub_metric == SubMetric.RELEVANCE][0]
        assert relevance.score < 0.6

    def test_quality_keyword_overlap(self, sample_spans):
        """Should compute correctness based on keyword overlap."""
        run = RunRecord(
            status=RunStatus.SUCCESS,
            final_output="The result is 37",
            expected_output="37",
        )
        evaluator = AnswerQualityEvaluator()
        results = evaluator.evaluate(run, sample_spans)
        correctness = [r for r in results if r.sub_metric == SubMetric.CORRECTNESS][0]
        assert correctness.score > 0.0


# ============================================================
# LatencyEvaluator Tests
# ============================================================


class TestLatencyEvaluator:
    def test_total_latency(self, sample_run, sample_spans):
        """Should report total latency correctly."""
        evaluator = LatencyEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        total = [r for r in results if r.sub_metric == SubMetric.TOTAL_LATENCY_MS][0]
        assert total.score == 5000
        assert total.passed is True  # 5000ms < 60s limit

    def test_avg_step_latency(self, sample_run, sample_spans):
        """Should compute average step latency."""
        evaluator = LatencyEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        avg = [r for r in results if r.sub_metric == SubMetric.AVG_STEP_LATENCY_MS][0]
        assert avg.score > 0

    def test_high_latency_fails(self, sample_spans):
        """Very high latency should fail budget check."""
        run = RunRecord(total_latency_ms=120000)  # 2 minutes
        evaluator = LatencyEvaluator()
        results = evaluator.evaluate(run, sample_spans)
        total = [r for r in results if r.sub_metric == SubMetric.TOTAL_LATENCY_MS][0]
        assert total.passed is False


# ============================================================
# TokenCostEvaluator Tests
# ============================================================


class TestTokenCostEvaluator:
    def test_token_breakdown(self, sample_run, sample_spans):
        """Should report token breakdown correctly."""
        evaluator = TokenCostEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)

        prompt = [r for r in results if r.sub_metric == SubMetric.PROMPT_TOKENS][0]
        assert prompt.score == 100

        completion = [r for r in results if r.sub_metric == SubMetric.COMPLETION_TOKENS][0]
        assert completion.score == 50

        total = [r for r in results if r.sub_metric == SubMetric.TOTAL_TOKENS][0]
        assert total.score == 150

    def test_cost_reporting(self, sample_run, sample_spans):
        """Should report cost."""
        evaluator = TokenCostEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        cost = [r for r in results if r.sub_metric == SubMetric.TOTAL_COST][0]
        assert cost.score == 0.0003
        assert cost.passed is True  # under $1

    def test_efficiency_computation(self, sample_run, sample_spans):
        """Should compute token efficiency."""
        evaluator = TokenCostEvaluator()
        results = evaluator.evaluate(sample_run, sample_spans)
        efficiency_results = [r for r in results if r.sub_metric is None]
        assert len(efficiency_results) >= 1
        efficiency = efficiency_results[0]
        assert 0 <= efficiency.score <= 1.0
