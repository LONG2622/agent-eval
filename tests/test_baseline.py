"""Tests for agent_eval.evaluation.baseline."""

from __future__ import annotations

import json
import time

import pytest

from agent_eval.evaluation.baseline import (
    _baselines_dir,
    compare_to_baseline,
    delete_baseline,
    list_baselines,
    load_baseline,
    save_baseline,
)
from agent_eval.evaluation.engine import EvaluationEngine
from agent_eval.trace.models import RunRecord, RunStatus, Span, SpanType, TokenUsage


def _make_run(
    run_id: str,
    *,
    status: RunStatus = RunStatus.SUCCESS,
    final_output: str = "",
    expected_output: str | None = None,
    total_latency_ms: int = 1000,
    tokens: TokenUsage | None = None,
) -> RunRecord:
    """Helper: create a minimal RunRecord."""
    return RunRecord(
        run_id=run_id,
        task_id=f"task_{run_id}",
        agent_name="test_agent",
        status=status,
        input_text="test question",
        final_output=final_output,
        expected_output=expected_output,
        total_steps=2,
        total_latency_ms=total_latency_ms,
        tokens=tokens or TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        total_cost=0.0001,
    )


def _make_span(trace_id: str, span_id: str, *, span_type: SpanType = SpanType.AGENT_STEP, is_success: bool = True) -> Span:
    """Helper: create a minimal Span."""
    return Span(
        span_id=span_id,
        trace_id=trace_id,
        span_type=span_type,
        step_index=0,
        name="test",
        is_success=is_success,
        latency_ms=100,
        tokens=TokenUsage(),
    )


def _seed_storage_with_runs(storage, run_specs: list[dict]) -> list[str]:
    """Seed storage with runs per spec. Each spec is a dict of RunRecord kwargs."""
    run_ids: list[str] = []
    for spec in run_specs:
        rid = spec.pop("run_id")
        run = _make_run(rid, **spec)
        storage.save_run(run)
        storage.append_span(_make_span(rid, f"{rid}_span"))
        run_ids.append(rid)
    return run_ids


# ============================================================
# Tests
# ============================================================


class TestSaveBaseline:
    def test_save_baseline_creates_file(self, patch_config, storage, tmp_output_dir):
        """Saving a baseline should create a JSON file under outputs/baselines/."""
        run_ids = _seed_storage_with_runs(
            storage,
            [
                {
                    "run_id": "r1",
                    "status": RunStatus.SUCCESS,
                    "final_output": "hello world",
                    "expected_output": "hello world",
                }
            ],
        )

        engine = EvaluationEngine(storage=storage)
        baseline_id = save_baseline(engine=engine, run_ids=run_ids, name="test-baseline", storage=storage)

        # File exists
        baselines_dir = _baselines_dir()
        baseline_path = baselines_dir / f"{baseline_id}.json"
        assert baseline_path.exists(), f"Baseline file not found at {baseline_path}"

        # Content is valid JSON
        with open(baseline_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["baseline_id"] == baseline_id
        assert data["name"] == "test-baseline"
        assert "summary" in data

    def test_save_baseline_custom_name_and_metadata(self, patch_config, storage):
        """Custom name, dataset_id, agent_name should all be preserved."""
        run_ids = _seed_storage_with_runs(
            storage,
            [{"run_id": "r1", "status": RunStatus.SUCCESS}],
        )
        engine = EvaluationEngine(storage=storage)
        bid = save_baseline(
            engine=engine,
            run_ids=run_ids,
            name="my-release",
            dataset_id="dataset_v3",
            agent_name="react-v2",
            storage=storage,
        )
        meta = load_baseline(bid)
        assert meta.name == "my-release"
        assert meta.dataset_id == "dataset_v3"
        assert meta.agent_name == "react-v2"
        assert "dataset_v3" not in meta.run_ids  # run_ids should only contain run IDs

    def test_save_baseline_all_runs_when_none(self, patch_config, storage):
        """When run_ids is None, ALL runs in storage should be evaluated."""
        _seed_storage_with_runs(
            storage,
            [
                {"run_id": "r1", "status": RunStatus.SUCCESS},
                {"run_id": "r2", "status": RunStatus.SUCCESS},
                {"run_id": "r3", "status": RunStatus.FAILED},
            ],
        )
        engine = EvaluationEngine(storage=storage)
        bid = save_baseline(engine=engine, storage=storage)
        meta = load_baseline(bid)
        assert len(meta.run_ids) == 3

    def test_save_baseline_empty_storage_raises(self, patch_config, storage):
        """Saving with zero runs should raise ValueError."""
        engine = EvaluationEngine(storage=storage)
        with pytest.raises(ValueError, match="No runs available"):
            save_baseline(engine=engine, storage=storage)


class TestLoadBaseline:
    def test_load_baseline_roundtrip(self, patch_config, storage):
        """Save then load — all fields must match."""
        run_ids = _seed_storage_with_runs(
            storage,
            [
                {"run_id": "r1", "status": RunStatus.SUCCESS, "final_output": "42", "expected_output": "42"},
                {"run_id": "r2", "status": RunStatus.SUCCESS, "final_output": "xyz", "expected_output": "xyz"},
            ],
        )
        engine = EvaluationEngine(storage=storage)
        original_bid = save_baseline(
            engine=engine, run_ids=run_ids, name="roundtrip", dataset_id="ds1", agent_name="ag1", storage=storage
        )

        loaded = load_baseline(original_bid)
        assert loaded.baseline_id == original_bid
        assert loaded.name == "roundtrip"
        assert loaded.dataset_id == "ds1"
        assert loaded.agent_name == "ag1"
        assert set(loaded.run_ids) == set(run_ids)
        assert "overall_success_rate" in loaded.summary
        assert "dimensions" in loaded.summary

    def test_load_missing_baseline_raises(self, patch_config, storage):
        """Loading a non-existent baseline must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_baseline("this_baseline_does_not_exist_12345")


class TestListBaselines:
    def test_list_baselines_multiple(self, patch_config, storage):
        """Saving 2 baselines should yield 2 entries sorted newest first."""
        run_ids = _seed_storage_with_runs(
            storage,
            [{"run_id": "r1", "status": RunStatus.SUCCESS}],
        )
        engine = EvaluationEngine(storage=storage)

        bid1 = save_baseline(engine=engine, run_ids=run_ids, name="first", storage=storage)
        time.sleep(0.01)  # ensure distinct mtime
        bid2 = save_baseline(engine=engine, run_ids=run_ids, name="second", storage=storage)

        baselines = list_baselines()
        assert len(baselines) == 2
        ids = [b.baseline_id for b in baselines]
        assert bid1 in ids
        assert bid2 in ids
        # Newest first
        assert baselines[0].baseline_id == bid2

    def test_list_baselines_empty(self, patch_config, storage):
        """Empty baseline dir → empty list."""
        assert list_baselines() == []


class TestDeleteBaseline:
    def test_delete_removes_file(self, patch_config, storage):
        run_ids = _seed_storage_with_runs(
            storage,
            [{"run_id": "r1", "status": RunStatus.SUCCESS}],
        )
        engine = EvaluationEngine(storage=storage)
        bid = save_baseline(engine=engine, run_ids=run_ids, name="del-me", storage=storage)
        assert delete_baseline(bid) is True
        assert delete_baseline("nonexistent") is False
        with pytest.raises(FileNotFoundError):
            load_baseline(bid)


class TestCompareToBaseline:
    def test_compare_no_regression(self, patch_config, storage):
        """Comparing baseline against the SAME runs should yield 0 deltas."""
        run_ids = _seed_storage_with_runs(
            storage,
            [
                {
                    "run_id": "r1",
                    "status": RunStatus.SUCCESS,
                    "final_output": "correct answer about apples and oranges",
                    "expected_output": "correct answer about apples and oranges",
                }
            ],
        )
        engine = EvaluationEngine(storage=storage)
        bid = save_baseline(engine=engine, run_ids=run_ids, name="same", storage=storage)

        result = compare_to_baseline(baseline_id=bid, current_run_ids=run_ids, engine=engine, storage=storage)
        # All dimension deltas must be (close to) zero
        for dim, d in result["dimension_deltas"].items():
            assert abs(d) < 1e-6, f"Expected delta 0 for dim {dim}, got {d}"
        assert result["overall_delta"] == 0.0
        assert result["regressions"] == []
        assert result["improvements"] == []
        assert "No regression" in result["note"]

    def test_compare_with_regression(self, patch_config, storage):
        """Baseline has ALL SUCCESS runs; current has one FAILED → negative delta."""
        base_run_ids = _seed_storage_with_runs(
            storage,
            [
                {
                    "run_id": "b1",
                    "status": RunStatus.SUCCESS,
                    "final_output": "correct apples oranges",
                    "expected_output": "correct apples oranges",
                },
                {
                    "run_id": "b2",
                    "status": RunStatus.SUCCESS,
                    "final_output": "correct foo bar",
                    "expected_output": "correct foo bar",
                },
            ],
        )
        engine = EvaluationEngine(storage=storage)
        bid = save_baseline(engine=engine, run_ids=base_run_ids, name="regress", storage=storage)

        # Now make a current run that is FAILED
        _seed_storage_with_runs(
            storage,
            [
                {"run_id": "c1", "status": RunStatus.FAILED},
                {
                    "run_id": "c2",
                    "status": RunStatus.SUCCESS,
                    "final_output": "correct apples oranges",
                    "expected_output": "correct apples oranges",
                },
            ],
        )
        current_ids = ["c1", "c2"]

        result = compare_to_baseline(
            baseline_id=bid, current_run_ids=current_ids, engine=engine, storage=storage
        )
        assert result["overall_delta"] < 0.0
        assert len(result["regressions"]) > 0
        assert any("regressed" in result["note"].lower() or "regress" in result["note"].lower() for _ in [1])
        assert result["note"] != ""

    def test_compare_with_improvement(self, patch_config, storage):
        """Baseline has some FAILED runs; current has all SUCCESS → positive delta."""
        base_run_ids = _seed_storage_with_runs(
            storage,
            [
                {"run_id": "b1", "status": RunStatus.FAILED},
                {
                    "run_id": "b2",
                    "status": RunStatus.SUCCESS,
                    "final_output": "hello world",
                    "expected_output": "hello world",
                },
            ],
        )
        engine = EvaluationEngine(storage=storage)
        bid = save_baseline(engine=engine, run_ids=base_run_ids, name="improve", storage=storage)

        # Current: both succeed
        _seed_storage_with_runs(
            storage,
            [
                {
                    "run_id": "c1",
                    "status": RunStatus.SUCCESS,
                    "final_output": "foo bar baz",
                    "expected_output": "foo bar baz",
                },
                {
                    "run_id": "c2",
                    "status": RunStatus.SUCCESS,
                    "final_output": "hello world",
                    "expected_output": "hello world",
                },
            ],
        )
        current_ids = ["c1", "c2"]

        result = compare_to_baseline(
            baseline_id=bid, current_run_ids=current_ids, engine=engine, storage=storage
        )
        assert result["overall_delta"] > 0.0
        assert len(result["improvements"]) > 0

    def test_compare_missing_baseline_raises(self, patch_config, storage):
        """Comparing against a non-existent baseline should raise FileNotFoundError."""
        engine = EvaluationEngine(storage=storage)
        with pytest.raises(FileNotFoundError):
            compare_to_baseline(baseline_id="no_such_baseline", engine=engine, storage=storage)

    def test_compare_empty_current_handled(self, patch_config, storage):
        """No current runs → graceful note, no crash."""
        run_ids = _seed_storage_with_runs(
            storage,
            [{"run_id": "b1", "status": RunStatus.SUCCESS}],
        )
        engine = EvaluationEngine(storage=storage)
        bid = save_baseline(engine=engine, run_ids=run_ids, name="empty-current", storage=storage)

        result = compare_to_baseline(
            baseline_id=bid, current_run_ids=[], engine=engine, storage=storage
        )
        assert result["current_run_count"] == 0
        assert "No current runs" in result["note"]
        assert result["overall_delta"] == 0.0
        assert result["regressions"] == []
        assert result["improvements"] == []
