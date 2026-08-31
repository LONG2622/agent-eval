"""Evaluation engine: orchestrates evaluators and aggregates metrics."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_eval.config import load_config
from agent_eval.evaluation.base import BaseEvaluator, EvalDimension, EvaluationResult, SubMetric
from agent_eval.evaluation.builtin import get_builtin_evaluator_instances
from agent_eval.logger import setup_logger
from agent_eval.trace import JSONLStorage, RunRecord, RunStatus

logger = setup_logger("agent_eval.evaluation.engine")


# ============================================================
# Aggregated Metrics Models
# ============================================================


@dataclass
class DimensionSummary:
    """Aggregate statistics for one evaluation dimension across a dataset."""

    dimension: EvalDimension
    count: int
    passed_count: int
    pass_rate: float
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "count": self.count,
            "passed_count": self.passed_count,
            "pass_rate": round(self.pass_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "median_score": round(self.median_score, 4),
            "min_score": round(self.min_score, 4),
            "max_score": round(self.max_score, 4),
        }


@dataclass
class BatchSummary:
    """Summary of evaluating a batch of runs."""

    total_runs: int
    evaluated_runs: int
    total_evaluation_results: int
    overall_success_rate: float
    overall_quality_score: float
    avg_latency_ms: float
    avg_cost: float
    total_cost: float
    total_tokens: int
    dimension_summaries: dict[str, DimensionSummary]
    top_failures: list[dict[str, Any]]
    top_successes: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "evaluated_runs": self.evaluated_runs,
            "total_evaluation_results": self.total_evaluation_results,
            "overall_success_rate": round(self.overall_success_rate, 4),
            "overall_quality_score": round(self.overall_quality_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_cost": round(self.avg_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "dimensions": {k: v.to_dict() for k, v in self.dimension_summaries.items()},
            "top_failures": self.top_failures,
            "top_successes": self.top_successes,
            "metadata": self.metadata,
        }


# ============================================================
# Engine
# ============================================================


class EvaluationEngine:
    """Runs a set of evaluators against runs, with aggregation + persistence."""

    def __init__(
        self,
        evaluators: list[BaseEvaluator] | None = None,
        storage: JSONLStorage | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        cfg = load_config()
        if evaluators is None:
            evaluators = get_builtin_evaluator_instances(cfg.evaluation.default_evaluators)
        self._evaluators = evaluators
        self._storage = storage or JSONLStorage()
        self._eval_dir = Path(output_dir or cfg.storage.eval_dir)
        self._eval_dir.mkdir(parents=True, exist_ok=True)
        # In-memory index run_id -> list[EvaluationResult]
        self._index: dict[str, list[EvaluationResult]] = {}

    # -------- Single Run --------

    def evaluate_run(self, run_id: str) -> list[EvaluationResult]:
        run = self._storage.load_run(run_id)
        if run is None:
            raise KeyError(f"Run {run_id} not found")
        spans = self._storage.load_spans(run_id)

        # Short-circuit: failed runs get zeroed scores across all dimensions
        if run.status == RunStatus.FAILED:
            all_results = self._failed_run_results(run)
            self._index[run_id] = all_results
            self._persist_run_results(run_id, all_results)
            return all_results

        all_results: list[EvaluationResult] = []
        for ev in self._evaluators:
            try:
                results = ev.evaluate(run, spans)
                all_results.extend(results)
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
                logger.error(f"Evaluator {ev.name} failed on run {run_id}: {e}")
        self._index[run_id] = all_results
        self._persist_run_results(run_id, all_results)
        return all_results

    @staticmethod
    def _failed_run_results(run: RunRecord) -> list[EvaluationResult]:
        """Generate zeroed evaluation results for a failed run."""
        dims = [
            (EvalDimension.SUCCESS_RATE, SubMetric.PASS),
            (EvalDimension.ANSWER_QUALITY, SubMetric.COMPLETENESS),
            (EvalDimension.TOOL_USAGE, None),
            (EvalDimension.LATENCY, SubMetric.TOTAL_LATENCY_MS),
            (EvalDimension.TOKEN_COST, SubMetric.TOTAL_TOKENS),
        ]
        results: list[EvaluationResult] = []
        for dim, sub in dims:
            results.append(
                EvaluationResult(
                    run_id=run.run_id,
                    evaluator="failed_run_shortcut",
                    dimension=dim,
                    sub_metric=sub,
                    score=0.0,
                    passed=False,
                    details={
                        "reason": "run_failed",
                        "error": run.error_message or "unknown",
                        "run_status": run.status.value,
                    },
                )
            )
            # Overall score for this dimension
            results.append(
                EvaluationResult(
                    run_id=run.run_id,
                    evaluator="failed_run_shortcut",
                    dimension=dim,
                    score=0.0,
                    passed=False,
                    details={"reason": "run_failed"},
                )
            )
        return results

    # -------- Batch --------

    def evaluate_runs(
        self, run_ids: Iterable[str], *, save_summary: bool = True
    ) -> tuple[dict[str, list[EvaluationResult]], BatchSummary]:
        per_run: dict[str, list[EvaluationResult]] = {}
        runs: list[RunRecord] = []
        for rid in run_ids:
            per_run[rid] = self.evaluate_run(rid)
            run = self._storage.load_run(rid)
            if run is not None:
                runs.append(run)
        summary = self.aggregate(runs, per_run)
        if save_summary:
            self._persist_summary(summary)
        return per_run, summary

    # -------- Aggregation --------

    def aggregate(
        self,
        runs: list[RunRecord],
        per_run_results: dict[str, list[EvaluationResult]],
    ) -> BatchSummary:
        # Flatten all results
        all_results: list[EvaluationResult] = []
        for rlist in per_run_results.values():
            all_results.extend(rlist)

        # Group by dimension (use overall per-dim per-run score)
        per_dimension_scores: dict[str, list[float]] = {}
        per_dimension_passed: dict[str, list[bool]] = {}
        for _rid, rlist in per_run_results.items():
            per_run_dim: dict[str, EvaluationResult] = {}
            for r in rlist:
                # Prefer the overall (sub_metric is None) result for the dim
                if r.sub_metric is None:
                    per_run_dim[r.dimension.value] = r
                else:
                    per_run_dim.setdefault(r.dimension.value, r)
            for dim, result in per_run_dim.items():
                per_dimension_scores.setdefault(dim, []).append(result.score)
                per_dimension_passed.setdefault(dim, []).append(bool(result.passed))

        dim_summaries: dict[str, DimensionSummary] = {}
        for dim, scores in per_dimension_scores.items():
            passed_list = per_dimension_passed.get(dim, [False] * len(scores))
            sorted_scores = sorted(scores)
            dim_summaries[dim] = DimensionSummary(
                dimension=EvalDimension(dim),
                count=len(scores),
                passed_count=sum(1 for p in passed_list if p),
                pass_rate=sum(1 for p in passed_list if p) / max(len(passed_list), 1),
                mean_score=statistics.mean(scores) if scores else 0.0,
                median_score=statistics.median(scores) if scores else 0.0,
                min_score=sorted_scores[0] if sorted_scores else 0.0,
                max_score=sorted_scores[-1] if sorted_scores else 0.0,
                scores=scores,
            )

        # Overall metrics from runs
        sr_dim = per_dimension_scores.get(EvalDimension.SUCCESS_RATE.value, [])
        q_dim = per_dimension_scores.get(EvalDimension.ANSWER_QUALITY.value, [])
        overall_success_rate = statistics.mean(sr_dim) if sr_dim else 0.0
        overall_quality = statistics.mean(q_dim) if q_dim else 0.0
        avg_latency = statistics.mean([r.total_latency_ms for r in runs]) if runs else 0.0
        avg_cost = statistics.mean([r.total_cost for r in runs]) if runs else 0.0
        total_cost = sum(r.total_cost for r in runs)
        total_tokens = sum(r.tokens.total_tokens for r in runs)

        # Top successes / failures
        scored = []
        for r in runs:
            rlist = per_run_results.get(r.run_id, [])
            overall = 0.0
            n = 0
            for er in rlist:
                if er.sub_metric is None:
                    overall += er.score
                    n += 1
            avg = overall / n if n else 0.0
            scored.append((avg, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_successes = [
            {"run_id": r.run_id, "task_id": r.task_id, "score": round(s, 4), "preview": (r.input_text[:60] + "...")}
            for s, r in scored[:5]
        ]
        top_failures = [
            {"run_id": r.run_id, "task_id": r.task_id, "score": round(s, 4), "preview": (r.input_text[:60] + "...")}
            for s, r in scored[-5:]
        ]

        return BatchSummary(
            total_runs=len(runs),
            evaluated_runs=len(per_run_results),
            total_evaluation_results=len(all_results),
            overall_success_rate=overall_success_rate,
            overall_quality_score=overall_quality,
            avg_latency_ms=avg_latency,
            avg_cost=avg_cost,
            total_cost=total_cost,
            total_tokens=total_tokens,
            dimension_summaries=dim_summaries,
            top_failures=list(reversed(top_failures)),
            top_successes=top_successes,
            metadata={"evaluators": [e.name for e in self._evaluators]},
        )

    # -------- Persistence --------

    def _persist_run_results(self, run_id: str, results: list[EvaluationResult]) -> None:
        path = self._eval_dir / f"eval_{run_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_storage_dict(), ensure_ascii=False) + "\n")

    def _persist_summary(self, summary: BatchSummary) -> Path:
        import time as _time

        path = self._eval_dir / f"batch_summary_{int(_time.time())}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Batch summary saved to {path}")
        return path

    # -------- Queries --------

    def get_run_results(self, run_id: str) -> list[EvaluationResult] | None:
        if run_id in self._index:
            return self._index[run_id]
        path = self._eval_dir / f"eval_{run_id}.jsonl"
        if not path.exists():
            return None
        results: list[EvaluationResult] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(EvaluationResult.model_validate_json(line))
        self._index[run_id] = results
        return results
