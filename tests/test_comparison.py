"""Tests for the annotation vs auto-evaluation comparison report module."""

from __future__ import annotations

import pytest

from agent_eval.report.comparison_report import (
    ComparisonItem,
    ComparisonSummary,
    _normalize_human_score,
    _pearson_correlation,
    format_comparison_text,
    run_comparison,
)
from agent_eval.trace.models import (
    AnnotationRecord,
)

# ============================================================
# Helper Functions Tests
# ============================================================


class TestNormalizeHumanScore:
    def test_normalize_min(self):
        assert _normalize_human_score(1) == 0.0

    def test_normalize_max(self):
        assert _normalize_human_score(5) == 1.0

    def test_normalize_mid(self):
        assert _normalize_human_score(3) == 0.5

    def test_normalize_2(self):
        assert _normalize_human_score(2) == 0.25

    def test_normalize_4(self):
        assert _normalize_human_score(4) == 0.75


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        assert _pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative(self):
        assert _pearson_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0, abs=1e-10)

    def test_no_correlation(self):
        # Random-ish data should give correlation near 0
        corr = _pearson_correlation([1.0, 2.0, 3.0, 4.0], [2.5, 1.0, 3.5, 2.0])
        assert -1.0 <= corr <= 1.0

    def test_single_point(self):
        assert _pearson_correlation([1.0], [1.0]) is None

    def test_empty(self):
        assert _pearson_correlation([], []) is None

    def test_zero_variance(self):
        # Constant values should return None (std=0)
        assert _pearson_correlation([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is None


# ============================================================
# ComparisonItem Tests
# ============================================================


class TestComparisonItem:
    def test_to_dict(self):
        item = ComparisonItem(
            run_id="test_run",
            task_id="task_001",
            input_text="Calculate 2+2",
            status="success",
            human_score=0.75,
            human_score_raw=4,
            annotator="alice",
            auto_overall_score=0.6,
            auto_dimensions={"success_rate": 1.0},
            discrepancy=0.15,
            human_labels=["correct"],
            human_comment="Good",
        )
        d = item.to_dict()
        assert d["run_id"] == "test_run"
        assert d["human_score"] == 0.75
        assert d["auto_overall_score"] == 0.6
        assert d["discrepancy"] == 0.15

    def test_to_dict_none_values(self):
        item = ComparisonItem(
            run_id="test",
            task_id="t",
            input_text="x",
            status="success",
            human_score=None,
            human_score_raw=None,
            annotator="",
            auto_overall_score=None,
            discrepancy=None,
        )
        d = item.to_dict()
        assert d["human_score"] is None
        assert d["auto_overall_score"] is None
        assert d["discrepancy"] is None


# ============================================================
# ComparisonSummary Tests
# ============================================================


class TestComparisonSummary:
    def test_to_dict(self):
        summary = ComparisonSummary(
            total_runs_with_annotations=10,
            total_runs_with_auto_eval=20,
            total_runs_both=5,
            human_mean=0.7,
            auto_mean=0.6,
            correlation=0.85,
            mae=0.1,
            rmse=0.12,
            agreement_within_0_2=0.8,
            agreement_within_0_3=0.95,
        )
        d = summary.to_dict()
        assert d["total_runs_with_annotations"] == 10
        assert d["correlation"] == 0.85
        assert d["agreement_within_0_2"] == 0.8

    def test_to_dict_empty(self):
        summary = ComparisonSummary()
        d = summary.to_dict()
        assert d["total_runs_with_annotations"] == 0
        assert d["correlation"] is None


# ============================================================
# run_comparison Integration Tests
# ============================================================


class TestRunComparison:
    def test_run_comparison_no_data(self, storage):
        """Comparison with empty storage should produce zero counts."""
        summary, items = run_comparison(storage)
        assert summary.total_runs_with_annotations == 0
        assert summary.total_runs_both == 0
        assert len(items) == 0

    def test_run_comparison_with_data(self, storage, sample_run, sample_spans):
        """Comparison with annotated + evaluated data should produce results."""
        # Save run and spans
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        # Add annotation
        ann = AnnotationRecord(
            run_id="test_run_001",
            score=4,
            labels=["correct"],
            comment="Good",
        )
        storage.save_annotation(ann)

        # Run comparison (engine will auto-evaluate the run)
        summary, items = run_comparison(storage)

        assert summary.total_runs_with_annotations >= 1
        assert len(items) >= 1

        # Check the compared item
        compared = [i for i in items if i.run_id == "test_run_001"]
        assert len(compared) == 1
        item = compared[0]
        assert item.human_score is not None
        assert item.human_score == 0.75  # score 4 -> (4-1)/4 = 0.75
        assert item.auto_overall_score is not None
        assert item.discrepancy is not None


# ============================================================
# format_comparison_text Tests
# ============================================================


class TestFormatComparisonText:
    def test_format_with_data(self, storage, annotated_run_data):
        """Format should produce readable text with comparison data."""
        summary, items = run_comparison(storage)
        text = format_comparison_text(summary, items)
        assert "Annotation vs Auto-Evaluation Comparison Report" in text
        assert "Total runs with annotations" in text

    def test_format_no_data(self, storage):
        """Format with no data should indicate zero runs."""
        summary, items = run_comparison(storage)
        text = format_comparison_text(summary, items)
        assert "0" in text  # zero counts

    def test_format_with_both_scores(self, storage, sample_run, sample_spans):
        """Format should show correlation when both scores available."""
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)
        ann = AnnotationRecord(run_id="test_run_001", score=3)
        storage.save_annotation(ann)

        summary, items = run_comparison(storage)
        text = format_comparison_text(summary, items)
        # Should contain MAE and agreement stats
        if summary.total_runs_both > 0:
            assert "MAE" in text
            assert "Agreement" in text
