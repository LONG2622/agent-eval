"""Comparison report: human annotations vs automatic evaluation scores.

This module compares human annotation scores (1-5 scale) with automatic
evaluation scores (0-1 scale) and generates a detailed comparison report
including correlation analysis, disagreement highlights, and summary stats.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_eval.evaluation.engine import EvaluationEngine
from agent_eval.trace import JSONLStorage


@dataclass
class ComparisonItem:
    """A single run's comparison between human and auto evaluation."""

    run_id: str
    task_id: str
    input_text: str
    status: str
    human_score: float | None  # 0-1 (normalized from 1-5)
    human_score_raw: int | None  # 1-5
    annotator: str
    auto_overall_score: float | None  # 0-1
    auto_dimensions: dict[str, float] = field(default_factory=dict)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    discrepancy: float | None = None  # absolute difference
    human_labels: list[str] = field(default_factory=list)
    human_comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "input_text": self.input_text[:100],
            "status": self.status,
            "human_score": round(self.human_score, 4) if self.human_score is not None else None,
            "human_score_raw": self.human_score_raw,
            "annotator": self.annotator,
            "auto_overall_score": round(self.auto_overall_score, 4) if self.auto_overall_score is not None else None,
            "auto_dimensions": {k: round(v, 4) for k, v in self.auto_dimensions.items()},
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "discrepancy": round(self.discrepancy, 4) if self.discrepancy is not None else None,
            "human_labels": self.human_labels,
            "human_comment": self.human_comment[:200],
        }


@dataclass
class ComparisonSummary:
    """Aggregated comparison results."""

    total_runs_with_annotations: int = 0
    total_runs_with_auto_eval: int = 0
    total_runs_both: int = 0  # both human and auto scores available
    human_mean: float | None = None
    auto_mean: float | None = None
    correlation: float | None = None  # Pearson correlation
    mae: float | None = None  # Mean Absolute Error
    rmse: float | None = None  # Root Mean Squared Error
    agreement_within_0_2: float | None = None  # % within 0.2 tolerance
    agreement_within_0_3: float | None = None  # % within 0.3 tolerance
    dimension_correlations: dict[str, float] = field(default_factory=dict)
    top_discrepancies: list[ComparisonItem] = field(default_factory=list)
    category_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_runs_with_annotations": self.total_runs_with_annotations,
            "total_runs_with_auto_eval": self.total_runs_with_auto_eval,
            "total_runs_both": self.total_runs_both,
            "human_mean": round(self.human_mean, 4) if self.human_mean is not None else None,
            "auto_mean": round(self.auto_mean, 4) if self.auto_mean is not None else None,
            "correlation": round(self.correlation, 4) if self.correlation is not None else None,
            "mae": round(self.mae, 4) if self.mae is not None else None,
            "rmse": round(self.rmse, 4) if self.rmse is not None else None,
            "agreement_within_0_2": round(self.agreement_within_0_2, 4) if self.agreement_within_0_2 is not None else None,
            "agreement_within_0_3": round(self.agreement_within_0_3, 4) if self.agreement_within_0_3 is not None else None,
            "dimension_correlations": {k: round(v, 4) for k, v in self.dimension_correlations.items()},
            "top_discrepancies": [item.to_dict() for item in self.top_discrepancies[:10]],
            "category_breakdown": self.category_breakdown,
        }


def _normalize_human_score(raw: int) -> float:
    """Convert 1-5 human score to 0-1 scale."""
    return (raw - 1) / 4.0


def _pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation between two lists."""
    n = len(x)
    if n < 2:
        return None
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def run_comparison(storage: JSONLStorage | None = None) -> tuple[ComparisonSummary, list[ComparisonItem]]:
    """Run comparison between human annotations and auto evaluation scores.

    Returns:
        (summary, items) where summary is the aggregated stats and items
        is the per-run comparison list.
    """
    _storage = storage or JSONLStorage()
    engine = EvaluationEngine(storage=_storage)

    # Load all annotations
    all_annotations = _storage.load_annotations()
    # Load all runs
    all_runs = _storage.list_runs()
    runs_by_id = {r.run_id: r for r in all_runs}

    # Build annotation index: run_id -> best (latest) annotation
    ann_by_run: dict[str, Any] = {}
    for ann in all_annotations:
        # Use the most recent annotation per run
        rid = ann.run_id
        if rid not in ann_by_run or ann.created_at > ann_by_run[rid].created_at:
            ann_by_run[rid] = ann

    items: list[ComparisonItem] = []
    human_scores: list[float] = []
    auto_scores: list[float] = []
    both_count = 0

    for run_id, run in runs_by_id.items():
        ann = ann_by_run.get(run_id)

        # Get auto evaluation results
        eval_results = engine.get_run_results(run_id)
        auto_overall = None
        auto_dims: dict[str, float] = {}

        if eval_results:
            # Compute overall auto score (average of dimension scores where sub_metric is None)
            dim_scores: dict[str, list[float]] = {}
            for er in eval_results:
                if er.sub_metric is None:
                    dim_scores.setdefault(er.dimension.value, []).append(er.score)
            for dim, scores in dim_scores.items():
                auto_dims[dim] = statistics.mean(scores) if scores else 0.0

            if auto_dims:
                auto_overall = statistics.mean(auto_dims.values())
            else:
                auto_overall = 0.0
        elif run.status.value == "success":
            # Trigger evaluation if not yet done
            try:
                eval_results = engine.evaluate_run(run_id)
                dim_scores: dict[str, list[float]] = {}
                for er in eval_results:
                    if er.sub_metric is None:
                        dim_scores.setdefault(er.dimension.value, []).append(er.score)
                for dim, scores in dim_scores.items():
                    auto_dims[dim] = statistics.mean(scores) if scores else 0.0
                if auto_dims:
                    auto_overall = statistics.mean(auto_dims.values())
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        human_norm = None
        human_raw = None
        annotator = ""
        labels: list[str] = []
        comment = ""

        if ann:
            human_raw = ann.score
            human_norm = _normalize_human_score(ann.score)
            annotator = ann.annotator
            labels = ann.labels or []
            comment = ann.comment or ""
            human_scores.append(human_norm)

        if auto_overall is not None:
            auto_scores.append(auto_overall)
            if ann:
                both_count += 1

        item = ComparisonItem(
            run_id=run_id,
            task_id=run.task_id,
            input_text=run.input_text or "",
            status=run.status.value,
            human_score=human_norm,
            human_score_raw=human_raw,
            annotator=annotator,
            auto_overall_score=auto_overall,
            auto_dimensions=auto_dims,
            dimension_scores=auto_dims,
            discrepancy=abs(human_norm - auto_overall) if (human_norm is not None and auto_overall is not None) else None,
            human_labels=labels,
            human_comment=comment,
        )
        items.append(item)

    # Compute summary
    summary = ComparisonSummary(
        total_runs_with_annotations=len(set(a.run_id for a in all_annotations)),
        total_runs_with_auto_eval=len([r for r in runs_by_id.values() if engine.get_run_results(r.run_id)]),
        total_runs_both=both_count,
    )

    if human_scores and auto_scores:
        # Align for correlation (only runs with both scores)
        paired_human = []
        paired_auto = []
        for item in items:
            if item.human_score is not None and item.auto_overall_score is not None:
                paired_human.append(item.human_score)
                paired_auto.append(item.auto_overall_score)

        if paired_human:
            summary.human_mean = statistics.mean(paired_human)
            summary.auto_mean = statistics.mean(paired_auto)
            summary.correlation = _pearson_correlation(paired_human, paired_auto)

            # MAE and RMSE
            abs_errors = [abs(h - a) for h, a in zip(paired_human, paired_auto, strict=False)]
            summary.mae = statistics.mean(abs_errors)
            summary.rmse = (statistics.mean([e ** 2 for e in abs_errors])) ** 0.5

            # Agreement within tolerance
            within_02 = sum(1 for e in abs_errors if e <= 0.2) / len(abs_errors)
            within_03 = sum(1 for e in abs_errors if e <= 0.3) / len(abs_errors)
            summary.agreement_within_0_2 = within_02
            summary.agreement_within_0_3 = within_03

            # Top discrepancies
            sorted_items = sorted(
                [i for i in items if i.discrepancy is not None],
                key=lambda x: x.discrepancy or 0,
                reverse=True,
            )
            summary.top_discrepancies = sorted_items[:10]

            # Category breakdown by human labels
            cat_breakdown: dict[str, dict[str, Any]] = {}
            for item in items:
                if item.human_score is not None and item.auto_overall_score is not None:
                    for label in item.human_labels:
                        if label not in cat_breakdown:
                            cat_breakdown[label] = {
                                "count": 0,
                                "human_scores": [],
                                "auto_scores": [],
                                "discrepancies": [],
                            }
                        cat_breakdown[label]["count"] += 1
                        cat_breakdown[label]["human_scores"].append(item.human_score)
                        cat_breakdown[label]["auto_scores"].append(item.auto_overall_score)
                        if item.discrepancy is not None:
                            cat_breakdown[label]["discrepancies"].append(item.discrepancy)

            for _label, data in cat_breakdown.items():
                data["human_mean"] = round(statistics.mean(data["human_scores"]), 4) if data["human_scores"] else 0
                data["auto_mean"] = round(statistics.mean(data["auto_scores"]), 4) if data["auto_scores"] else 0
                data["mean_discrepancy"] = round(statistics.mean(data["discrepancies"]), 4) if data["discrepancies"] else 0
                del data["human_scores"]
                del data["auto_scores"]
                del data["discrepancies"]

            summary.category_breakdown = cat_breakdown

    return summary, items


def generate_comparison_report(
    storage: JSONLStorage | None = None,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a full comparison report and save as JSON."""
    summary, items = run_comparison(storage)

    if output_dir is None:
        from agent_eval.config import load_config
        cfg = load_config()
        output_dir = Path(cfg.storage.output_dir) / "reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary.to_dict(),
        "items": [item.to_dict() for item in items],
    }

    path = output_dir / f"comparison_report_{int(time.time())}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return path


def format_comparison_text(summary: ComparisonSummary, items: list[ComparisonItem]) -> str:
    """Format comparison results as readable text for CLI output."""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 Annotation vs Auto-Evaluation Comparison Report")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Total runs with annotations:  {summary.total_runs_with_annotations}")
    lines.append(f"Total runs with auto-eval:   {summary.total_runs_with_auto_eval}")
    lines.append(f"Runs with BOTH scores:       {summary.total_runs_both}")
    lines.append("")

    if summary.total_runs_both == 0:
        lines.append("⚠️  No runs have both human annotations and auto-evaluation scores.")
        lines.append("   Create some annotations via the web UI first.")
        return "\n".join(lines)

    lines.append("--- Overall Statistics ---")
    lines.append(f"  Human score mean:     {summary.human_mean:.4f}" if summary.human_mean is not None else "  Human score mean:     N/A")
    lines.append(f"  Auto score mean:      {summary.auto_mean:.4f}" if summary.auto_mean is not None else "  Auto score mean:      N/A")
    lines.append(f"  Pearson correlation:  {summary.correlation:.4f}" if summary.correlation is not None else "  Pearson correlation:  N/A")
    lines.append("")

    lines.append("--- Error Metrics ---")
    lines.append(f"  MAE:                  {summary.mae:.4f}" if summary.mae is not None else "  MAE:                  N/A")
    lines.append(f"  RMSE:                 {summary.rmse:.4f}" if summary.rmse is not None else "  RMSE:                 N/A")
    lines.append(f"  Agreement (±0.2):    {summary.agreement_within_0_2:.1%}" if summary.agreement_within_0_2 is not None else "  Agreement (±0.2):    N/A")
    lines.append(f"  Agreement (±0.3):    {summary.agreement_within_0_3:.1%}" if summary.agreement_within_0_3 is not None else "  Agreement (±0.3):    N/A")
    lines.append("")

    if summary.top_discrepancies:
        lines.append("--- Top Discrepancies ---")
        lines.append("-" * 60)
        for item in summary.top_discrepancies[:10]:
            status_icon = "✅" if item.status == "success" else "❌"
            lines.append(f"  {status_icon} {item.run_id[:12]}  disc={item.discrepancy:.3f}  human={item.human_score:.2f}  auto={item.auto_overall_score:.2f}" if item.discrepancy is not None and item.human_score is not None and item.auto_overall_score is not None else f"  {status_icon} {item.run_id[:12]}  N/A")
            lines.append(f"      Task: {item.input_text[:60]}")
            if item.human_comment:
                lines.append(f"      Comment: {item.human_comment[:80]}")
            lines.append("")

    if summary.category_breakdown:
        lines.append("--- Breakdown by Human Labels ---")
        lines.append("-" * 60)
        for label, data in sorted(summary.category_breakdown.items(), key=lambda x: x[1]["count"], reverse=True):
            lines.append(f"  [{label}] count={data['count']}  human_avg={data['human_mean']:.3f}  auto_avg={data['auto_mean']:.3f}  mean_diff={data['mean_discrepancy']:.3f}")

    return "\n".join(lines)
