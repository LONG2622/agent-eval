"""Tests for JSONLStorage - run/span/annotation CRUD operations."""

from __future__ import annotations

import json

import pytest

from agent_eval.trace.models import (
    AnnotationRecord,
    RunRecord,
    RunStatus,
    Span,
    SpanType,
    TokenUsage,
)
from agent_eval.trace.storage import JSONLStorage


class TestJSONLStorage:
    def test_init_creates_directories(self, tmp_output_dir):
        """Storage should create trace/run/annotation dirs on init."""
        storage = JSONLStorage(
            trace_dir=tmp_output_dir / "traces",
            run_dir=tmp_output_dir / "runs",
            annotation_dir=tmp_output_dir / "annotations",
        )
        assert (tmp_output_dir / "traces").exists()
        assert (tmp_output_dir / "runs").exists()
        assert (tmp_output_dir / "annotations").exists()

    # ---- Span Tests ----

    def test_append_and_load_span(self, storage):
        """Append a span, then load it back."""
        span = Span(
            trace_id="run_001",
            span_type=SpanType.LLM_CALL,
            step_index=0,
            name="test_model",
            input_data={"prompt": "hello"},
            latency_ms=100,
        )
        storage.append_span(span)
        loaded = storage.load_spans("run_001")
        assert len(loaded) == 1
        assert loaded[0].trace_id == "run_001"
        assert loaded[0].name == "test_model"
        assert loaded[0].input_data == {"prompt": "hello"}

    def test_append_spans_batch(self, storage):
        """append_spans should handle multiple spans."""
        spans = [
            Span(trace_id="run_001", span_type=SpanType.AGENT_STEP, step_index=0),
            Span(trace_id="run_001", span_type=SpanType.LLM_CALL, step_index=1),
            Span(trace_id="run_002", span_type=SpanType.TOOL_CALL, step_index=0),
        ]
        storage.append_spans(spans)

        loaded1 = storage.load_spans("run_001")
        loaded2 = storage.load_spans("run_002")
        assert len(loaded1) == 2
        assert len(loaded2) == 1

    def test_load_nonexistent_trace(self, storage):
        """Loading spans for nonexistent trace should return empty list."""
        loaded = storage.load_spans("nonexistent")
        assert loaded == []

    def test_spans_sorted_by_step_index(self, storage):
        """Spans should be returned sorted by step_index."""
        spans = [
            Span(trace_id="run_001", span_type=SpanType.AGENT_STEP, step_index=3),
            Span(trace_id="run_001", span_type=SpanType.LLM_CALL, step_index=1),
            Span(trace_id="run_001", span_type=SpanType.TOOL_CALL, step_index=2),
        ]
        storage.append_spans(spans)
        loaded = storage.load_spans("run_001")
        assert [s.step_index for s in loaded] == [1, 2, 3]

    # ---- Run Tests ----

    def test_save_and_load_run(self, storage):
        """Save a run, then load it back."""
        run = RunRecord(
            run_id="run_001",
            status=RunStatus.SUCCESS,
            final_output="42",
        )
        storage.save_run(run)
        loaded = storage.load_run("run_001")
        assert loaded is not None
        assert loaded.run_id == "run_001"
        assert loaded.status == RunStatus.SUCCESS
        assert loaded.final_output == "42"

    def test_load_nonexistent_run(self, storage):
        """Loading nonexistent run should return None."""
        loaded = storage.load_run("nonexistent")
        assert loaded is None

    def test_list_runs(self, storage):
        """list_runs should return all runs sorted by started_at."""
        r1 = RunRecord(run_id="run_001", task_id="task_a")
        r2 = RunRecord(run_id="run_002", task_id="task_b")
        r3 = RunRecord(run_id="run_003", task_id="task_a")
        storage.save_run(r1)
        storage.save_run(r2)
        storage.save_run(r3)

        all_runs = storage.list_runs()
        assert len(all_runs) == 3

        filtered = storage.list_runs(task_id="task_a")
        assert len(filtered) == 2
        assert all(r.task_id == "task_a" for r in filtered)

    def test_update_existing_run(self, storage):
        """Saving a run with same ID should update it."""
        run = RunRecord(run_id="run_001", status=RunStatus.PENDING)
        storage.save_run(run)
        run.mark_finished(RunStatus.SUCCESS)
        storage.save_run(run)
        loaded = storage.load_run("run_001")
        assert loaded.status == RunStatus.SUCCESS

    # ---- Annotation Tests ----

    def test_save_and_load_annotation(self, storage):
        """Save annotation, then load by run_id."""
        ann = AnnotationRecord(
            run_id="run_001",
            score=4,
            labels=["correct"],
            comment="Great!",
        )
        storage.save_annotation(ann)
        loaded = storage.load_annotations("run_001")
        assert len(loaded) == 1
        assert loaded[0].score == 4
        assert loaded[0].comment == "Great!"

    def test_load_all_annotations(self, storage):
        """load_annotations() without arg should return all annotations."""
        a1 = AnnotationRecord(run_id="run_001", score=3)
        a2 = AnnotationRecord(run_id="run_002", score=5)
        storage.save_annotation(a1)
        storage.save_annotation(a2)
        all_anns = storage.load_annotations()
        assert len(all_anns) == 2

    def test_delete_annotation(self, storage):
        """delete_annotation should remove the annotation."""
        ann = AnnotationRecord(run_id="run_001", score=3)
        storage.save_annotation(ann)
        result = storage.delete_annotation(ann.annotation_id)
        assert result is True
        loaded = storage.load_annotations("run_001")
        assert len(loaded) == 0

    def test_delete_nonexistent_annotation(self, storage):
        """Deleting nonexistent annotation should return False."""
        result = storage.delete_annotation("nonexistent_id")
        assert result is False

    def test_multiple_annotations_same_run(self, storage):
        """Multiple annotations for same run should all be loadable."""
        a1 = AnnotationRecord(run_id="run_001", score=3, annotator="alice")
        a2 = AnnotationRecord(run_id="run_001", score=5, annotator="bob")
        storage.save_annotation(a1)
        storage.save_annotation(a2)
        loaded = storage.load_annotations("run_001")
        assert len(loaded) == 2
        annotators = {a.annotator for a in loaded}
        assert annotators == {"alice", "bob"}

    # ---- Persistence Tests ----

    def test_persistence_across_instances(self, tmp_output_dir):
        """Data should persist across different Storage instances."""
        trace_dir = tmp_output_dir / "traces"
        run_dir = tmp_output_dir / "runs"
        ann_dir = tmp_output_dir / "annotations"

        # First instance: write data
        s1 = JSONLStorage(trace_dir=trace_dir, run_dir=run_dir, annotation_dir=ann_dir)
        run = RunRecord(run_id="persist_test", status=RunStatus.SUCCESS)
        s1.save_run(run)
        span = Span(trace_id="persist_test", span_type=SpanType.AGENT_STEP, step_index=0)
        s1.append_span(span)
        ann = AnnotationRecord(run_id="persist_test", score=4)
        s1.save_annotation(ann)

        # Second instance: read data
        s2 = JSONLStorage(trace_dir=trace_dir, run_dir=run_dir, annotation_dir=ann_dir)
        loaded_run = s2.load_run("persist_test")
        loaded_spans = s2.load_spans("persist_test")
        loaded_anns = s2.load_annotations("persist_test")

        assert loaded_run is not None
        assert loaded_run.status == RunStatus.SUCCESS
        assert len(loaded_spans) == 1
        assert len(loaded_anns) == 1


class TestStorageWithRealData:
    """Integration-style tests using the sample conftest fixtures."""

    def test_full_round_trip(self, annotated_run_data, storage):
        """Verify the annotated_run_data fixture produced correct data."""
        run, spans, ann = annotated_run_data

        loaded_run = storage.load_run("test_run_001")
        assert loaded_run is not None
        assert loaded_run.status == RunStatus.SUCCESS

        loaded_spans = storage.load_spans("test_run_001")
        assert len(loaded_spans) == 4

        loaded_anns = storage.load_annotations("test_run_001")
        assert len(loaded_anns) == 1
        assert loaded_anns[0].score == 4
        assert loaded_anns[0].labels == ["correct", "complete"]

    def test_list_runs_with_data(self, annotated_run_data, storage):
        """list_runs should include the test run."""
        runs = storage.list_runs()
        assert len(runs) >= 1
        run_ids = {r.run_id for r in runs}
        assert "test_run_001" in run_ids