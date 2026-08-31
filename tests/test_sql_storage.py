"""Tests for SQLiteStorage - schema init, CRUD round-trips, upsert semantics,
filters, aggregation queries, invalid data handling, and JSONL -> SQLite migration."""

from __future__ import annotations

import sqlite3

import pytest
from pydantic import ValidationError

from agent_eval.evaluation.base import EvalDimension, EvaluationResult, SubMetric
from agent_eval.trace.models import RunRecord, RunStatus, Span, SpanType, TokenUsage
from agent_eval.trace.sql_storage import SQLiteStorage
from agent_eval.trace.storage import JSONLStorage


@pytest.fixture
def sqlite_storage(patch_config, tmp_path):
    """SQLiteStorage backed by a throwaway db file."""
    storage = SQLiteStorage(db_path=tmp_path / "test.db")
    yield storage
    storage.close()


_UNSET = object()


def _make_eval_result(
    run_id: str,
    *,
    sub_metric: SubMetric | None = None,
    score: float = 0.9,
    passed: bool | None = True,
    details=_UNSET,
) -> EvaluationResult:
    return EvaluationResult(
        run_id=run_id,
        evaluator="test_evaluator",
        dimension=EvalDimension.ANSWER_QUALITY,
        sub_metric=sub_metric,
        score=score,
        passed=passed,
        details=details if details is not _UNSET else {"reason": "unit_test"},
    )


class TestSQLiteSchema:
    def test_init_creates_db_file_and_tables(self, sqlite_storage, tmp_path):
        """Constructor should create the db file plus all four core tables."""
        assert (tmp_path / "test.db").exists()
        tables = {
            row["name"]
            for row in sqlite_storage._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"tasks", "runs", "spans", "evaluations"} <= tables

    def test_save_run_twice_still_one_row(self, sqlite_storage, sample_run):
        """INSERT OR REPLACE means saving twice must not duplicate rows."""
        sqlite_storage.save_run(sample_run)
        sqlite_storage.save_run(sample_run)
        count = sqlite_storage._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == 1


class TestRunRoundTrip:
    def test_save_and_load_run(self, sqlite_storage, sample_run):
        sqlite_storage.save_run(sample_run)
        loaded = sqlite_storage.load_run("test_run_001")
        assert loaded is not None
        assert loaded.run_id == sample_run.run_id
        assert loaded.task_id == sample_run.task_id
        assert loaded.agent_name == sample_run.agent_name
        assert loaded.status == RunStatus.SUCCESS
        assert loaded.input_text == sample_run.input_text
        assert loaded.final_output == sample_run.final_output
        assert loaded.total_steps == sample_run.total_steps
        assert loaded.total_latency_ms == sample_run.total_latency_ms
        assert loaded.tokens == sample_run.tokens
        assert loaded.total_cost == pytest.approx(sample_run.total_cost)
        assert loaded.expected_output == sample_run.expected_output
        assert loaded.started_at == sample_run.started_at

    def test_load_nonexistent_run_returns_none(self, sqlite_storage):
        assert sqlite_storage.load_run("ghost_run") is None

    def test_save_run_twice_is_upsert(self, sqlite_storage):
        """Same run_id overwrites the previous row (status update visible)."""
        run = RunRecord(run_id="upsert_run", status=RunStatus.PENDING)
        sqlite_storage.save_run(run)
        run.mark_finished(RunStatus.SUCCESS)
        run.final_output = "done"
        sqlite_storage.save_run(run)

        loaded = sqlite_storage.load_run("upsert_run")
        assert loaded.status == RunStatus.SUCCESS
        assert loaded.final_output == "done"
        assert loaded.finished_at is not None
        assert len(sqlite_storage.list_runs()) == 1


class TestSpanRoundTrip:
    def test_save_and_load_spans(self, sqlite_storage, sample_run, sample_spans):
        # FK constraint: spans.trace_id -> runs.id, so save the run first
        sqlite_storage.save_run(sample_run)
        sqlite_storage.append_spans(sample_spans)

        loaded = sqlite_storage.load_spans("test_run_001")
        assert len(loaded) == 4
        # Returned ordered by step_index
        assert [s.step_index for s in loaded] == [0, 1, 2, 3]

        first, orig = loaded[0], sample_spans[0]
        assert first.span_id == orig.span_id
        assert first.trace_id == orig.trace_id
        assert first.span_type == SpanType.AGENT_STEP
        assert first.name == orig.name
        assert first.input_data == orig.input_data
        assert first.output_data == orig.output_data
        assert first.tokens == orig.tokens
        assert first.cost == pytest.approx(orig.cost)
        assert first.latency_ms == orig.latency_ms
        assert first.is_success is True
        assert first.created_at == orig.created_at

    def test_failed_span_round_trip(self, sqlite_storage):
        run = RunRecord(run_id="fail_run", status=RunStatus.FAILED)
        sqlite_storage.save_run(run)
        span = Span(
            trace_id="fail_run",
            span_type=SpanType.TOOL_CALL,
            name="search",
            is_success=False,
            exception="TimeoutError: boom",
        )
        sqlite_storage.append_span(span)

        loaded = sqlite_storage.load_spans("fail_run")
        assert len(loaded) == 1
        assert loaded[0].is_success is False
        assert loaded[0].exception == "TimeoutError: boom"

    def test_load_spans_for_unknown_trace_returns_empty(self, sqlite_storage):
        assert sqlite_storage.load_spans("ghost_trace") == []


class TestEvaluationRoundTrip:
    def test_save_and_load_evaluations(self, sqlite_storage, sample_run):
        sqlite_storage.save_run(sample_run)
        results = [
            _make_eval_result("test_run_001", sub_metric=SubMetric.CORRECTNESS, score=0.85,
                              passed=True, details={"matched": 3}),
            _make_eval_result("test_run_001", sub_metric=None, score=0.72, passed=True),
            _make_eval_result("test_run_001", sub_metric=SubMetric.COMPLETENESS, score=0.2,
                              passed=False, details=None),
            _make_eval_result("test_run_001", sub_metric=None, score=0.0, passed=None),
        ]
        sqlite_storage.save_evaluations(results)

        loaded = sqlite_storage.load_evaluations("test_run_001")
        assert len(loaded) == 4
        assert sorted(r.score for r in loaded) == sorted([0.85, 0.72, 0.2, 0.0])

        correctness = next(r for r in loaded if r.sub_metric == SubMetric.CORRECTNESS)
        assert correctness.run_id == "test_run_001"
        assert correctness.evaluator == "test_evaluator"
        assert correctness.dimension == EvalDimension.ANSWER_QUALITY
        assert correctness.score == pytest.approx(0.85)
        assert correctness.passed is True
        assert correctness.details == {"matched": 3}

        overall = [r for r in loaded if r.sub_metric is None]
        assert len(overall) == 2
        assert next(r for r in overall if r.passed is None).passed is None

        failed = next(r for r in loaded if r.passed is False)
        assert failed.score == pytest.approx(0.2)
        assert failed.details is None
        assert all(r.is_human is False for r in loaded)

    def test_save_single_evaluation(self, sqlite_storage, sample_run):
        sqlite_storage.save_run(sample_run)
        sqlite_storage.save_evaluation(_make_eval_result("test_run_001", score=0.5, passed=False))
        loaded = sqlite_storage.load_evaluations("test_run_001")
        assert len(loaded) == 1
        assert loaded[0].passed is False

    def test_orphan_evaluation_rejected_by_foreign_key(self, sqlite_storage):
        """Evaluations reference runs(id); saving one for an unknown run must fail."""
        orphan = _make_eval_result("ghost_run")
        with pytest.raises(sqlite3.IntegrityError):
            sqlite_storage.save_evaluation(orphan)


class TestListRunsFilters:
    def test_filter_by_task_and_status(self, sqlite_storage):
        r1 = RunRecord(run_id="run_a1", task_id="task_a", status=RunStatus.SUCCESS,
                       started_at="2026-01-01T00:00:00Z")
        r2 = RunRecord(run_id="run_a2", task_id="task_a", status=RunStatus.FAILED,
                       started_at="2026-01-02T00:00:00Z")
        r3 = RunRecord(run_id="run_b1", task_id="task_b", status=RunStatus.SUCCESS,
                       started_at="2026-01-03T00:00:00Z")
        for r in (r1, r2, r3):
            sqlite_storage.save_run(r)

        assert len(sqlite_storage.list_runs()) == 3

        by_task = sqlite_storage.list_runs(task_id="task_a")
        assert {r.run_id for r in by_task} == {"run_a1", "run_a2"}

        by_status = sqlite_storage.list_runs(status="success")
        assert {r.run_id for r in by_status} == {"run_a1", "run_b1"}

        combined = sqlite_storage.list_runs(task_id="task_a", status="failed")
        assert {r.run_id for r in combined} == {"run_a2"}

    def test_limit_and_ordering(self, sqlite_storage):
        r1 = RunRecord(run_id="old_run", started_at="2026-01-01T00:00:00Z")
        r2 = RunRecord(run_id="new_run", started_at="2026-01-02T00:00:00Z")
        sqlite_storage.save_run(r1)
        sqlite_storage.save_run(r2)

        limited = sqlite_storage.list_runs(limit=1)
        assert len(limited) == 1
        assert limited[0].run_id == "new_run"  # newest first


class TestAggregates:
    def test_empty_db_returns_empty_dict(self, sqlite_storage):
        assert sqlite_storage.query_aggregates() == {}

    def test_aggregate_metrics(self, sqlite_storage):
        ok = RunRecord(run_id="ok_run", status=RunStatus.SUCCESS,
                       total_latency_ms=1000,
                       tokens=TokenUsage(total_tokens=100), total_cost=0.002)
        bad = RunRecord(run_id="bad_run", status=RunStatus.FAILED,
                        total_latency_ms=3000,
                        tokens=TokenUsage(total_tokens=300), total_cost=0.004)
        sqlite_storage.save_run(ok)
        sqlite_storage.save_run(bad)

        agg = sqlite_storage.query_aggregates()
        assert agg["total_runs"] == 2
        assert agg["success_rate"] == pytest.approx(0.5)
        assert agg["avg_latency_ms"] == pytest.approx(2000.0)
        assert agg["total_cost"] == pytest.approx(0.006)
        assert agg["avg_tokens"] == pytest.approx(200.0)


class TestInvalidDataHandling:
    def test_run_model_rejects_invalid_status(self):
        """Invalid data is rejected at the pydantic model layer before storage."""
        with pytest.raises(ValidationError):
            RunRecord(run_id="bad_run", status="not-a-status")

    def test_save_run_rejects_non_model_object(self, sqlite_storage):
        """save_run expects a RunRecord; a plain dict has no attributes to read."""
        with pytest.raises(AttributeError):
            sqlite_storage.save_run({"run_id": "dict_run", "status": "success"})


class TestMigrateFromJsonl:
    def test_migrate_imports_runs_and_spans(self, patch_config, tmp_path, sample_run, sample_spans):
        # Build a JSONL source: 2 runs + 4 spans each
        trace_dir = tmp_path / "migrate_traces"
        run_dir = tmp_path / "migrate_runs"
        jsonl = JSONLStorage(
            trace_dir=trace_dir,
            run_dir=run_dir,
            annotation_dir=tmp_path / "migrate_annotations",
        )
        second_run = RunRecord(
            run_id="test_run_002",
            task_id="task_002",
            status=RunStatus.SUCCESS,
            final_output="ok",
        )
        jsonl.save_run(sample_run)
        jsonl.save_run(second_run)
        jsonl.append_spans(sample_spans)
        jsonl.append_spans([s.model_copy(update={"trace_id": "test_run_002"}) for s in sample_spans])

        sqlite_storage = SQLiteStorage(db_path=tmp_path / "migrated.db")
        try:
            counts = sqlite_storage.migrate_from_jsonl(trace_dir, run_dir)
            assert counts == {"runs": 2, "spans": 8}

            loaded = sqlite_storage.load_run("test_run_002")
            assert loaded is not None
            assert loaded.status == RunStatus.SUCCESS
            assert sqlite_storage.load_run("test_run_001") is not None
            assert len(sqlite_storage.load_spans("test_run_001")) == 4
            assert len(sqlite_storage.load_spans("test_run_002")) == 4
        finally:
            sqlite_storage.close()

    def test_migrate_skips_invalid_jsonl_lines(self, patch_config, tmp_path, sample_run, sample_spans):
        trace_dir = tmp_path / "migrate_traces_bad"
        run_dir = tmp_path / "migrate_runs_bad"
        jsonl = JSONLStorage(
            trace_dir=trace_dir,
            run_dir=run_dir,
            annotation_dir=tmp_path / "migrate_annotations_bad",
        )
        jsonl.save_run(sample_run)
        jsonl.append_spans(sample_spans)

        # Corrupt run lines: invalid JSON and schema-invalid status
        with open(run_dir / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write("this is not json\n")
            f.write('{"status": "bogus"}\n')
        # Corrupt span file: garbage line only
        (trace_dir / "broken_trace.jsonl").write_text("garbage line\n", encoding="utf-8")

        sqlite_storage = SQLiteStorage(db_path=tmp_path / "migrated_bad.db")
        try:
            counts = sqlite_storage.migrate_from_jsonl(trace_dir, run_dir)
            # Only the single valid run + its 4 valid spans survive
            assert counts == {"runs": 1, "spans": 4}
        finally:
            sqlite_storage.close()
