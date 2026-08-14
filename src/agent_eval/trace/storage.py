"""JSONL-based storage backend for traces and runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from agent_eval.config import load_config
from agent_eval.trace.models import AnnotationRecord, RunRecord, Span

logger = logging.getLogger("agent_eval.trace.storage")


class JSONLStorage:
    """Appends spans and run records to newline-delimited JSON files.

    Layout:
        <trace_dir>/<trace_id>.jsonl    - one Span per line
        <run_dir>/runs.jsonl            - one RunRecord per line (append-only)
        <annotation_dir>/annotations.jsonl - one AnnotationRecord per line
    """

    def __init__(self, trace_dir: str | Path | None = None, run_dir: str | Path | None = None, annotation_dir: str | Path | None = None) -> None:
        cfg = load_config()
        self._trace_dir = Path(trace_dir or cfg.storage.trace_dir)
        self._run_dir = Path(run_dir or cfg.storage.run_dir)
        self._annotation_dir = Path(annotation_dir or cfg.storage.output_dir) / "annotations"
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._annotation_dir.mkdir(parents=True, exist_ok=True)
        self._runs_index: dict[str, RunRecord] = {}
        self._annotations_index: list[AnnotationRecord] = []
        self._load_runs_index()
        self._load_annotations_index()

    # ---- Spans ----

    def _trace_file(self, trace_id: str) -> Path:
        return self._trace_dir / f"{trace_id}.jsonl"

    def append_span(self, span: Span) -> None:
        path = self._trace_file(span.trace_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(span.to_storage_dict(), ensure_ascii=False) + "\n")

    def append_spans(self, spans: Iterable[Span]) -> None:
        # Group by trace_id for fewer file opens
        grouped: dict[str, list[Span]] = {}
        for s in spans:
            grouped.setdefault(s.trace_id, []).append(s)
        for trace_id, trace_spans in grouped.items():
            path = self._trace_file(trace_id)
            with open(path, "a", encoding="utf-8") as f:
                for s in trace_spans:
                    f.write(json.dumps(s.to_storage_dict(), ensure_ascii=False) + "\n")

    def load_spans(self, trace_id: str) -> list[Span]:
        path = self._trace_file(trace_id)
        if not path.exists():
            return []
        spans: list[Span] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    spans.append(Span.model_validate_json(line))
                except Exception as e:
                    logger.warning(f"Skipping invalid span line in {path}: {e}")
        spans.sort(key=lambda s: (s.step_index, s.created_at))
        return spans

    # ---- Runs ----

    def _runs_file(self) -> Path:
        return self._run_dir / "runs.jsonl"

    def _load_runs_index(self) -> None:
        path = self._runs_file()
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = RunRecord.model_validate_json(line)
                    self._runs_index[record.run_id] = record
                except Exception as e:
                    logger.warning(f"Skipping invalid run line: {e}")

    def save_run(self, run: RunRecord) -> None:
        self._runs_index[run.run_id] = run
        path = self._runs_file()
        # Rewrite full runs file for simplicity (small dataset in MVP)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in self._runs_index.values():
                f.write(json.dumps(r.to_storage_dict(), ensure_ascii=False) + "\n")
        tmp.replace(path)

    def load_run(self, run_id: str) -> RunRecord | None:
        if run_id in self._runs_index:
            return self._runs_index[run_id]
        # Reload from disk if not found in memory (handles cross-instance writes)
        self._load_runs_index()
        return self._runs_index.get(run_id)

    def list_runs(self, *, task_id: str | None = None) -> list[RunRecord]:
        # Reload from disk to pick up any writes from other instances
        self._load_runs_index()
        runs = list(self._runs_index.values())
        if task_id:
            runs = [r for r in runs if r.task_id == task_id]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs

    # ---- Annotations ----

    def _annotations_file(self) -> Path:
        return self._annotation_dir / "annotations.jsonl"

    def _load_annotations_index(self) -> None:
        path = self._annotations_file()
        if not path.exists():
            return
        self._annotations_index = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = AnnotationRecord.model_validate_json(line)
                    self._annotations_index.append(record)
                except Exception as e:
                    logger.warning(f"Skipping invalid annotation line: {e}")

    def save_annotation(self, annotation: AnnotationRecord) -> None:
        self._annotations_index.append(annotation)
        path = self._annotations_file()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for a in self._annotations_index:
                f.write(json.dumps(a.to_storage_dict(), ensure_ascii=False) + "\n")
        tmp.replace(path)

    def load_annotations(self, run_id: str | None = None) -> list[AnnotationRecord]:
        self._load_annotations_index()
        if run_id:
            return [a for a in self._annotations_index if a.run_id == run_id]
        return list(self._annotations_index)

    def delete_annotation(self, annotation_id: str) -> bool:
        self._load_annotations_index()
        original_len = len(self._annotations_index)
        self._annotations_index = [a for a in self._annotations_index if a.annotation_id != annotation_id]
        if len(self._annotations_index) < original_len:
            path = self._annotations_file()
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for a in self._annotations_index:
                    f.write(json.dumps(a.to_storage_dict(), ensure_ascii=False) + "\n")
            tmp.replace(path)
            return True
        return False
