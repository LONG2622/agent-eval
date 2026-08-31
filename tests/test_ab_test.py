"""Tests for ABTestRunner offline logic: label building, paired significance test,
overall/dimension comparison, task details, and the compare() pipeline with a
stubbed TaskRunner.  No network calls are made."""

from __future__ import annotations

import pytest

from agent_eval.evaluation.ab_test import ABTestRunner, ABTestSummary
from agent_eval.evaluation.base import EvalDimension
from agent_eval.evaluation.engine import BatchSummary, DimensionSummary
from agent_eval.task.runner import RunOutcome, TaskDataset, TaskItem
from agent_eval.trace.models import RunRecord, RunStatus, TokenUsage


def _make_outcomes(latencies: list[int], tokens: list[int]) -> list[RunOutcome]:
    """Build RunOutcomes with controlled latency/token numbers."""
    outcomes: list[RunOutcome] = []
    for i, (lat, tok) in enumerate(zip(latencies, tokens, strict=False)):
        task = TaskItem(task_id=f"t{i}", input=f"question {i}", expected_output="42")
        run = RunRecord(
            run_id=f"run_{i}",
            task_id=task.task_id,
            agent_name="stub",
            status=RunStatus.SUCCESS,
            input_text=task.input,
            final_output="42",
            total_latency_ms=lat,
            tokens=TokenUsage(prompt_tokens=tok // 2, completion_tokens=tok - tok // 2,
                              total_tokens=tok),
            total_cost=0.0001,
        )
        outcomes.append(RunOutcome(task=task, output="42", run=run))
    return outcomes


def _dimension_summary(dim: str, mean: float, pass_rate: float) -> DimensionSummary:
    return DimensionSummary(
        dimension=EvalDimension(dim),
        count=3,
        passed_count=round(pass_rate * 3),
        pass_rate=pass_rate,
        mean_score=mean,
        median_score=mean,
        min_score=max(0.0, mean - 0.1),
        max_score=min(1.0, mean + 0.1),
        scores=[mean] * 3,
    )


def _batch_summary(*, success_rate: float, quality: float, avg_latency_ms: float,
                   total_tokens: int, dims: dict) -> BatchSummary:
    return BatchSummary(
        total_runs=3,
        evaluated_runs=3,
        total_evaluation_results=15,
        overall_success_rate=success_rate,
        overall_quality_score=quality,
        avg_latency_ms=avg_latency_ms,
        avg_cost=0.0003,
        total_cost=0.0009,
        total_tokens=total_tokens,
        dimension_summaries=dims,
        top_failures=[],
        top_successes=[],
        metadata={"evaluators": ["success_rate", "answer_quality_keyword"]},
    )


class StubTaskRunner:
    """Stands in for TaskRunner.run_batch; returns canned outcomes per agent label."""

    def __init__(self, outcomes_by_label: dict[str, list[RunOutcome]],
                 summaries_by_label: dict[str, BatchSummary | None]) -> None:
        self._outcomes = outcomes_by_label
        self._summaries = summaries_by_label
        self.calls: list[str] = []

    def run_batch(self, dataset, **kwargs):
        prefix = kwargs["agent_name_prefix"]
        self.calls.append(prefix)
        for label, outcomes in self._outcomes.items():
            if prefix.startswith(label):
                return outcomes, self._summaries.get(label)
        raise AssertionError(f"unexpected run_batch call: {prefix}")


# ============================================================
# Label building
# ============================================================


class TestMakeLabel:
    def test_label_with_model(self):
        label = ABTestRunner._make_label({"agent_type": "react", "model": "org/gpt-4o-mini"}, "A")
        assert label == "react-gpt-4o-mini"

    def test_label_without_model(self):
        assert ABTestRunner._make_label({"agent_type": "react"}, "A") == "react"

    def test_label_defaults_to_react(self):
        assert ABTestRunner._make_label({}, "A") == "react"

    def test_runner_injection(self):
        """Providing a task_runner must avoid constructing the real TaskRunner."""
        stub = StubTaskRunner({}, {})
        assert ABTestRunner(task_runner=stub)._runner is stub


# ============================================================
# Paired significance test
# ============================================================


class TestComputeSignificance:
    def test_clear_difference_is_significant(self):
        a = _make_outcomes([1000, 1200, 900, 1100, 950], [100, 120, 90, 110, 95])
        b = _make_outcomes([5000, 5000, 5000, 5000, 5000], [1000, 1000, 1000, 1000, 1000])
        sig = ABTestRunner._compute_significance(a, b)

        assert sig["latency"]["significant"] is True
        assert sig["latency"]["mean_diff"] == pytest.approx(-3970.0)
        assert sig["latency"]["n"] == 5
        assert sig["tokens"]["significant"] is True
        assert sig["tokens"]["mean_diff"] == pytest.approx(-897.0)

    def test_identical_samples_not_significant(self):
        a = _make_outcomes([1000, 1100, 1050], [100, 110, 105])
        b = _make_outcomes([1000, 1100, 1050], [100, 110, 105])
        sig = ABTestRunner._compute_significance(a, b)

        assert sig["latency"]["significant"] is False
        assert sig["latency"]["t_statistic"] == 0.0
        assert sig["tokens"]["significant"] is False

    def test_too_few_samples(self):
        a = _make_outcomes([100, 200], [10, 20])
        b = _make_outcomes([300, 400], [30, 40])
        sig = ABTestRunner._compute_significance(a, b)
        assert "note" in sig

    def test_mismatched_lengths(self):
        a = _make_outcomes([100, 200, 300], [10, 20, 30])
        b = _make_outcomes([300, 400], [30, 40])
        sig = ABTestRunner._compute_significance(a, b)
        assert "note" in sig


# ============================================================
# Overall metric comparison
# ============================================================


class TestCompareOverall:
    def test_delta_computation(self):
        summary_a = _batch_summary(success_rate=1.0, quality=0.9, avg_latency_ms=1000.0,
                                   total_tokens=330, dims={})
        summary_b = _batch_summary(success_rate=0.5, quality=0.5, avg_latency_ms=5000.0,
                                   total_tokens=3000, dims={})
        overall = ABTestRunner._compare_overall(summary_a, summary_b)

        assert overall["agent_a"]["success_rate"] == 1.0
        assert overall["agent_b"]["quality_score"] == 0.5
        delta = overall["delta"]
        assert delta["success_rate"] == pytest.approx(0.5)
        assert delta["quality_score"] == pytest.approx(0.4)
        assert delta["avg_latency_ratio"] == pytest.approx(0.2)
        assert delta["total_tokens"] == -2670

    def test_none_summaries_give_empty_dict(self):
        assert ABTestRunner._compare_overall(None, None) == {}


# ============================================================
# Per-dimension comparison
# ============================================================


class TestCompareDimensions:
    def test_shared_dimensions_compared(self):
        summary_a = _batch_summary(success_rate=1.0, quality=0.9, avg_latency_ms=1000.0,
                                   total_tokens=330,
                                   dims={"answer_quality": _dimension_summary("answer_quality", 0.9, 1.0)})
        summary_b = _batch_summary(success_rate=0.5, quality=0.5, avg_latency_ms=5000.0,
                                   total_tokens=3000,
                                   dims={"answer_quality": _dimension_summary("answer_quality", 0.5, 0.6667)})
        result = ABTestRunner._compare_dimensions(summary_a, summary_b)

        entry = result["answer_quality"]
        assert entry["agent_a"]["mean"] == pytest.approx(0.9)
        assert entry["agent_b"]["mean"] == pytest.approx(0.5)
        assert entry["mean_delta"] == pytest.approx(0.4)
        assert entry["pass_rate_delta"] == pytest.approx(0.3333)

    def test_dimension_in_only_one_summary_is_skipped(self):
        summary_a = _batch_summary(success_rate=1.0, quality=0.9, avg_latency_ms=1000.0,
                                   total_tokens=330,
                                   dims={"latency": _dimension_summary("latency", 0.8, 0.9)})
        summary_b = _batch_summary(success_rate=0.5, quality=0.5, avg_latency_ms=5000.0,
                                   total_tokens=3000, dims={})
        assert ABTestRunner._compare_dimensions(summary_a, summary_b) == {}


# ============================================================
# Per-task details
# ============================================================


class TestBuildTaskDetails:
    def test_details_deltas(self):
        a = _make_outcomes([900, 1000, 1100], [100, 110, 120])
        b = _make_outcomes([4900, 4800, 5300], [900, 1000, 1100])
        details = ABTestRunner._build_task_details(a, b)

        assert len(details) == 3
        first = details[0]
        assert first["task_id"] == "t0"
        assert first["latency_delta_ms"] == -4000
        assert first["token_delta"] == -800
        assert first["agent_a"]["status"] == "success"
        assert first["agent_b"]["output_preview"] == "42"


# ============================================================
# ABTestSummary serialization
# ============================================================


class TestABTestSummary:
    def test_to_dict_caps_task_details_at_20(self):
        summary = ABTestSummary(
            agent_a_name="a",
            agent_b_name="b",
            dataset_name="d",
            total_tasks=25,
            task_details=[{"task_id": f"t{i}"} for i in range(25)],
        )
        d = summary.to_dict()
        assert len(d["task_details"]) == 20
        assert d["total_tasks"] == 25
        assert d["agent_a"] == "a"
        assert d["agent_b"] == "b"
        assert d["dataset"] == "d"


# ============================================================
# compare() pipeline with a stubbed runner
# ============================================================


class TestComparePipeline:
    def test_compare_builds_full_summary(self):
        dataset = TaskDataset.from_list(
            [
                {"task_id": f"t{i}", "input": f"question {i}", "expected_output": "42"}
                for i in range(3)
            ],
            name="toy",
        )
        # Agent A: faster and better. Latency diffs (-4000, -3800, -4200) and
        # token diffs (-800, -890, -980) both yield |t| >> 2 -> significant.
        outcomes_a = _make_outcomes([900, 1000, 1100], [100, 110, 120])
        outcomes_b = _make_outcomes([4900, 4800, 5300], [900, 1000, 1100])
        summary_a = _batch_summary(
            success_rate=1.0, quality=0.9, avg_latency_ms=1000.0, total_tokens=330,
            dims={"answer_quality": _dimension_summary("answer_quality", 0.9, 1.0)},
        )
        summary_b = _batch_summary(
            success_rate=0.5, quality=0.5, avg_latency_ms=5000.0, total_tokens=3000,
            dims={"answer_quality": _dimension_summary("answer_quality", 0.5, 0.6667)},
        )
        stub = StubTaskRunner(
            {"agent-a": outcomes_a, "agent-b": outcomes_b},
            {"agent-a": summary_a, "agent-b": summary_b},
        )
        runner = ABTestRunner(task_runner=stub)

        summary = runner.compare(
            dataset,
            agent_a={"agent_type": "react", "model": "stub-a"},
            agent_b={"agent_type": "react", "model": "stub-b"},
            label_a="agent-a",
            label_b="agent-b",
        )

        # Both agents were run, in order, with the expected name prefixes
        assert stub.calls == ["agent-a-", "agent-b-"]

        assert isinstance(summary, ABTestSummary)
        assert summary.agent_a_name == "agent-a"
        assert summary.agent_b_name == "agent-b"
        assert summary.dataset_name == "toy"
        assert summary.total_tasks == 3

        # Statistical significance: A is faster/leaner
        assert summary.significance["latency"]["significant"] is True
        assert summary.significance["latency"]["mean_diff"] == pytest.approx(-4000.0)
        assert summary.significance["tokens"]["significant"] is True

        # Overall + per-dimension deltas favour A
        assert summary.overall["delta"]["quality_score"] == pytest.approx(0.4)
        assert summary.overall["delta"]["avg_latency_ratio"] == pytest.approx(0.2)
        assert summary.dimensions["answer_quality"]["mean_delta"] == pytest.approx(0.4)

        # Per-task details + serialization
        assert len(summary.task_details) == 3
        assert summary.task_details[0]["latency_delta_ms"] == -4000
        assert summary.task_details[0]["token_delta"] == -800

        d = summary.to_dict()
        assert d["agent_a"] == "agent-a"
        assert d["dataset"] == "toy"
        assert len(d["task_details"]) == 3
