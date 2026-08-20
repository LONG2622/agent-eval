"""Tests for EvaluationEngine - single run evaluation and batch aggregation."""

from __future__ import annotations

import pytest

from agent_eval.evaluation.builtin import (
    AnswerQualityEvaluator,
    LatencyEvaluator,
    SuccessRateEvaluator,
    TokenCostEvaluator,
    ToolUsageEvaluator,
)
from agent_eval.evaluation.engine import EvaluationEngine
from agent_eval.trace.models import RunRecord, RunStatus, Span, SpanType


@pytest.fixture
def engine(storage):
    """Create an EvaluationEngine with default evaluators."""
    return EvaluationEngine(storage=storage)


class TestEvaluationEngine:
    def test_evaluate_run_success(self, engine, storage, sample_run, sample_spans):
        """Evaluating a successful run should produce results."""
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        results = engine.evaluate_run("test_run_001")
        assert len(results) > 0

        # Should have results from all evaluators
        evaluator_names = {r.evaluator for r in results}
        assert "success_rate" in evaluator_names
        assert "tool_usage" in evaluator_names
        assert "latency" in evaluator_names
        assert "token_cost" in evaluator_names
        assert "answer_quality_keyword" in evaluator_names

    def test_evaluate_run_not_found(self, engine):
        """Evaluating nonexistent run should raise KeyError."""
        with pytest.raises(KeyError):
            engine.evaluate_run("nonexistent_run")

    def test_evaluate_failed_run(self, engine, storage):
        """Failed runs should get zeroed shortcut results."""
        run = RunRecord(
            run_id="fail_run",
            status=RunStatus.FAILED,
            error_message="LLM timeout",
        )
        storage.save_run(run)

        results = engine.evaluate_run("fail_run")
        assert len(results) > 0
        # All scores should be 0.0
        for r in results:
            assert r.score == 0.0
            assert r.passed is False

    def test_evaluate_runs_batch(self, engine, storage, sample_run, sample_spans):
        """evaluate_runs should return per-run results and batch summary."""
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        per_run, summary = engine.evaluate_runs(["test_run_001"])
        assert "test_run_001" in per_run
        assert summary.total_runs == 1
        assert summary.evaluated_runs == 1
        assert summary.total_evaluation_results > 0
        assert 0 <= summary.overall_success_rate <= 1.0
        assert summary.avg_latency_ms >= 0

    def test_aggregate(self, engine, storage, sample_run, sample_spans):
        """aggregate should compute dimension summaries."""
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        per_run, summary = engine.evaluate_runs(["test_run_001"])

        # Check dimension summaries exist
        assert "success_rate" in summary.dimension_summaries
        assert "tool_usage" in summary.dimension_summaries
        assert "latency" in summary.dimension_summaries
        assert "token_cost" in summary.dimension_summaries
        assert "answer_quality" in summary.dimension_summaries

        # Each dimension summary should have required fields
        for dim, ds in summary.dimension_summaries.items():
            assert ds.count > 0
            assert 0 <= ds.pass_rate <= 1.0
            assert ds.mean_score >= 0

    def test_get_run_results(self, engine, storage, sample_run, sample_spans):
        """get_run_results should return cached or persisted results."""
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        # First evaluate
        engine.evaluate_run("test_run_001")

        # Then get
        results = engine.get_run_results("test_run_001")
        assert results is not None
        assert len(results) > 0

    def test_get_run_results_not_found(self, engine):
        """get_run_results for nonexistent run should return None."""
        results = engine.get_run_results("nonexistent")
        assert results is None

    def test_multiple_runs_batch(self, engine, storage, sample_spans):
        """Batch evaluation with multiple runs."""
        runs = []
        for i in range(3):
            run = RunRecord(
                run_id=f"run_{i}",
                task_id=f"task_{i}",
                status=RunStatus.SUCCESS,
                final_output=str(42 + i),
                total_steps=2,
                total_latency_ms=1000 * (i + 1),
            )
            runs.append(run)
            storage.save_run(run)
            storage.append_spans([
                span.model_copy(update={"trace_id": f"run_{i}"})
                for span in sample_spans
            ])

        run_ids = [r.run_id for r in runs]
        per_run, summary = engine.evaluate_runs(run_ids)
        assert len(per_run) == 3
        assert summary.total_runs == 3
        assert summary.evaluated_runs == 3
        assert summary.total_evaluation_results > 0

    def test_batch_summary_top_successes(self, engine, storage, sample_spans):
        """Top successes/failures should be populated."""
        # Create runs with different quality
        good_run = RunRecord(
            run_id="good_run",
            status=RunStatus.SUCCESS,
            final_output="42",
            input_text="What is 40+2?",
        )
        bad_run = RunRecord(
            run_id="bad_run",
            status=RunStatus.FAILED,
            error_message="timeout",
            input_text="Complex task that failed",
        )

        storage.save_run(good_run)
        storage.save_run(bad_run)

        good_spans = [s.model_copy(update={"trace_id": "good_run"}) for s in sample_spans]
        storage.append_spans(good_spans)

        engine.evaluate_runs(["good_run", "bad_run"])

        _, summary = engine.evaluate_runs(["good_run", "bad_run"])
        assert len(summary.top_successes) > 0
        assert len(summary.top_failures) > 0


class TestDimensionSummary:
    def test_to_dict(self, storage):
        """DimensionSummary.to_dict should be serializable."""
        engine = EvaluationEngine(storage=storage)

        run = RunRecord(run_id="dim_test", status=RunStatus.SUCCESS, final_output="ok")
        storage.save_run(run)
        storage.append_spans([
            Span(
                trace_id="dim_test",
                span_type="agent_step",
                step_index=0,
                name="test",
                latency_ms=100,
            )
        ])
        per_run, summary = engine.evaluate_runs(["dim_test"])
        for dim, ds in summary.dimension_summaries.items():
            d = ds.to_dict()
            assert "dimension" in d
            assert "count" in d
            assert "pass_rate" in d
            assert "mean_score" in d


class TestBatchSummary:
    def test_to_dict(self, engine, storage, sample_run, sample_spans):
        """BatchSummary.to_dict should produce full dict."""
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)
        _, summary = engine.evaluate_runs(["test_run_001"])
        d = summary.to_dict()
        assert "total_runs" in d
        assert "dimensions" in d
        assert "top_successes" in d
        assert "top_failures" in d
        assert "metadata" in d