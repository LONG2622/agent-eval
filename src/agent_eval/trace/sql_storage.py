# 8.12
"""SQLite-based storage backend for runs, spans, and evaluation results.

Provides persistent, queryable storage with four core tables:
    tasks       – task definitions
    runs        – agent execution records
    spans       – individual trace events
    evaluations – per-run evaluation scores

Supports CRUD, aggregation queries, and a JSONL→SQLite migration helper.
Fully compatible with the JSONLStorage interface so the two backends can be
switched at runtime via ``configs/default.yaml → storage.backend``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_eval.config import load_config
from agent_eval.trace.models import RunRecord, Span

if TYPE_CHECKING:
    from agent_eval.evaluation.base import EvaluationResult

logger = logging.getLogger("agent_eval.trace.sql_storage")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    input TEXT NOT NULL,
    expected_output TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_id TEXT,
    agent_name TEXT,
    agent_config TEXT,
    status TEXT NOT NULL,
    input_text TEXT,
    final_output TEXT,
    total_steps INTEGER DEFAULT 0,
    total_latency_ms INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    error_message TEXT,
    metadata TEXT,
    expected_output TEXT,
    ground_truth TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);

CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    span_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    span_type TEXT NOT NULL,
    step_index INTEGER DEFAULT 0,
    name TEXT,
    input_data TEXT,
    output_data TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    latency_ms INTEGER DEFAULT 0,
    is_success INTEGER DEFAULT 1,
    exception TEXT,
    metadata TEXT,
    created_at TEXT,
    FOREIGN KEY (trace_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_type ON spans(span_type);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    evaluator TEXT NOT NULL,
    dimension TEXT NOT NULL,
    sub_metric TEXT,
    score REAL DEFAULT 0.0,
    passed INTEGER,
    details TEXT,
    is_human INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_eval_run_id ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_dimension ON evaluations(dimension);
"""


class SQLiteStorage:
    """SQLite storage backend.  Drop-in replacement for :class:`JSONLStorage`."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        cfg = load_config()
        output_dir = Path(cfg.storage.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = Path(db_path or output_dir / "agent_eval.db")
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Spans
    # ------------------------------------------------------------------
    def append_span(self, span: Span) -> None:
        self._conn.execute(
            "INSERT INTO spans (span_id, trace_id, parent_span_id, span_type, step_index, name, "
            "input_data, output_data, prompt_tokens, completion_tokens, total_tokens, cost, "
            "latency_ms, is_success, exception, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                span.span_id,
                span.trace_id,
                span.parent_span_id,
                span.span_type.value,
                span.step_index,
                span.name,
                json.dumps(span.input_data, ensure_ascii=False),
                json.dumps(span.output_data, ensure_ascii=False),
                span.tokens.prompt_tokens,
                span.tokens.completion_tokens,
                span.tokens.total_tokens,
                span.cost,
                span.latency_ms,
                1 if span.is_success else 0,
                span.exception,
                json.dumps(span.metadata, ensure_ascii=False),
                span.created_at,
            ),
        )
        self._conn.commit()

    def append_spans(self, spans: Iterable[Span]) -> None:
        rows = []
        for s in spans:
            rows.append(
                (
                    s.span_id, s.trace_id, s.parent_span_id, s.span_type.value,
                    s.step_index, s.name,
                    json.dumps(s.input_data, ensure_ascii=False),
                    json.dumps(s.output_data, ensure_ascii=False),
                    s.tokens.prompt_tokens, s.tokens.completion_tokens, s.tokens.total_tokens,
                    s.cost, s.latency_ms,
                    1 if s.is_success else 0, s.exception,
                    json.dumps(s.metadata, ensure_ascii=False), s.created_at,
                )
            )
        self._conn.executemany(
            "INSERT INTO spans (span_id, trace_id, parent_span_id, span_type, step_index, name, "
            "input_data, output_data, prompt_tokens, completion_tokens, total_tokens, cost, "
            "latency_ms, is_success, exception, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def load_spans(self, trace_id: str) -> list[Span]:
        rows = self._conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY step_index, created_at",
            (trace_id,),
        ).fetchall()
        spans: list[Span] = []
        for r in rows:
            spans.append(
                Span(
                    span_id=r["span_id"],
                    trace_id=r["trace_id"],
                    parent_span_id=r["parent_span_id"],
                    span_type=r["span_type"],
                    step_index=r["step_index"],
                    name=r["name"],
                    input_data=json.loads(r["input_data"] or "{}"),
                    output_data=json.loads(r["output_data"] or "{}"),
                    tokens={
                        "prompt_tokens": r["prompt_tokens"],
                        "completion_tokens": r["completion_tokens"],
                        "total_tokens": r["total_tokens"],
                    },
                    cost=r["cost"],
                    latency_ms=r["latency_ms"],
                    is_success=bool(r["is_success"]),
                    exception=r["exception"],
                    metadata=json.loads(r["metadata"] or "{}"),
                    created_at=r["created_at"],
                )
            )
        return spans

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def save_run(self, run: RunRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (id, task_id, agent_name, agent_config, status, "
            "input_text, final_output, total_steps, total_latency_ms, "
            "prompt_tokens, completion_tokens, total_tokens, total_cost, "
            "error_message, metadata, expected_output, ground_truth, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.run_id, run.task_id, run.agent_name,
                json.dumps(run.agent_config, ensure_ascii=False),
                run.status.value,
                run.input_text, run.final_output,
                run.total_steps, run.total_latency_ms,
                run.tokens.prompt_tokens, run.tokens.completion_tokens, run.tokens.total_tokens,
                run.total_cost,
                run.error_message,
                json.dumps(run.metadata, ensure_ascii=False),
                run.expected_output,
                json.dumps(run.ground_truth, ensure_ascii=False) if run.ground_truth else None,
                run.started_at, run.finished_at,
            ),
        )
        self._conn.commit()

    def load_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_runs(
        self,
        *,
        task_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        sql = "SELECT * FROM runs WHERE 1=1"
        params: list[Any] = []
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_run(r) for r in rows]

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["id"],
            task_id=row["task_id"] or "",
            agent_name=row["agent_name"] or "",
            agent_config=json.loads(row["agent_config"] or "{}"),
            status=row["status"],
            input_text=row["input_text"] or "",
            final_output=row["final_output"],
            total_steps=row["total_steps"] or 0,
            total_latency_ms=row["total_latency_ms"] or 0,
            tokens={
                "prompt_tokens": row["prompt_tokens"] or 0,
                "completion_tokens": row["completion_tokens"] or 0,
                "total_tokens": row["total_tokens"] or 0,
            },
            total_cost=row["total_cost"] or 0.0,
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"] or "{}"),
            expected_output=row["expected_output"],
            ground_truth=json.loads(row["ground_truth"] or "null") if row["ground_truth"] else None,
            started_at=row["started_at"] or "",
            finished_at=row["finished_at"],
        )

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------
    def save_evaluation(self, result: EvaluationResult) -> None:

        self._conn.execute(
            "INSERT INTO evaluations (run_id, evaluator, dimension, sub_metric, score, "
            "passed, details, is_human) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.run_id,
                result.evaluator,
                result.dimension.value,
                result.sub_metric.value if result.sub_metric else None,
                result.score,
                1 if result.passed else 0 if result.passed is not None else None,
                json.dumps(result.details, ensure_ascii=False) if result.details else None,
                1 if result.is_human else 0,
            ),
        )
        self._conn.commit()

    def save_evaluations(self, results: Iterable[EvaluationResult]) -> None:

        rows = [
            (
                r.run_id, r.evaluator, r.dimension.value,
                r.sub_metric.value if r.sub_metric else None,
                r.score,
                1 if r.passed else 0 if r.passed is not None else None,
                json.dumps(r.details, ensure_ascii=False) if r.details else None,
                1 if r.is_human else 0,
            )
            for r in results
        ]
        self._conn.executemany(
            "INSERT INTO evaluations (run_id, evaluator, dimension, sub_metric, score, "
            "passed, details, is_human) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def load_evaluations(self, run_id: str) -> list[EvaluationResult]:
        from agent_eval.evaluation.base import EvaluationResult  # noqa: F811

        rows = self._conn.execute(
            "SELECT * FROM evaluations WHERE run_id = ? ORDER BY dimension, sub_metric",
            (run_id,),
        ).fetchall()
        results: list[EvaluationResult] = []
        for r in rows:
            results.append(
                EvaluationResult(
                    run_id=r["run_id"],
                    evaluator=r["evaluator"],
                    dimension=r["dimension"],
                    sub_metric=r["sub_metric"],
                    score=r["score"],
                    passed=bool(r["passed"]) if r["passed"] is not None else None,
                    details=json.loads(r["details"] or "null"),
                    is_human=bool(r["is_human"]),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Aggregation queries
    # ------------------------------------------------------------------
    def query_aggregates(self) -> dict[str, Any]:
        """Return high-level aggregate metrics across all runs."""
        total_runs = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        if total_runs == 0:
            return {}
        row = self._conn.execute(
            "SELECT "
            "AVG(CASE WHEN status='success' THEN 1.0 ELSE 0.0 END) as success_rate, "
            "AVG(total_latency_ms) as avg_latency, "
            "SUM(total_cost) as total_cost, "
            "AVG(total_tokens) as avg_tokens "
            "FROM runs"
        ).fetchone()
        return {
            "total_runs": total_runs,
            "success_rate": round(row["success_rate"] or 0.0, 4),
            "avg_latency_ms": round(row["avg_latency"] or 0.0, 1),
            "total_cost": round(row["total_cost"] or 0.0, 6),
            "avg_tokens": round(row["avg_tokens"] or 0.0, 0),
        }

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------
    def migrate_from_jsonl(self, trace_dir: str | Path, run_dir: str | Path) -> dict[str, int]:
        """Import all JSONL trace/run files into SQLite.  Returns counts of imported rows."""
        trace_dir = Path(trace_dir)
        run_dir = Path(run_dir)
        counts = {"runs": 0, "spans": 0}

        # Import runs
        runs_file = run_dir / "runs.jsonl"
        if runs_file.exists():
            with open(runs_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        run = RunRecord.model_validate_json(line)
                        self.save_run(run)
                        counts["runs"] += 1
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Skipping run line: {e}")

        # Import spans (one file per trace_id)
        if trace_dir.exists():
            for trace_file in trace_dir.glob("*.jsonl"):
                if trace_file.name == "runs.jsonl":
                    continue
                spans: list[Span] = []
                with open(trace_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            spans.append(Span.model_validate_json(line))
                        except (ValueError, TypeError, KeyError) as e:
                            logger.warning(f"Skipping span line in {trace_file}: {e}")
                if spans:
                    self.append_spans(spans)
                    counts["spans"] += len(spans)

        logger.info(f"Migrated {counts['runs']} runs and {counts['spans']} spans to SQLite")
        return counts

    def close(self) -> None:
        self._conn.close()
