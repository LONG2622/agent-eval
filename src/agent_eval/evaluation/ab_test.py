# 8.12
"""A/B testing engine: compare two agent configurations on the same dataset.

Usage::

    runner = ABTestRunner()
    summary = runner.compare(
        dataset,
        agent_a={"agent_type": "react", "model": "model-A"},
        agent_b={"agent_type": "react", "model": "model-B"},
    )

The runner executes both agents independently, evaluates all runs, and produces
an :class:`ABTestSummary` with per-dimension comparisons, aggregate metrics,
and a paired t-test for statistical significance.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from agent_eval.evaluation.engine import BatchSummary
from agent_eval.task.runner import RunOutcome, TaskDataset, TaskRunner


@dataclass
class ABTestSummary:
    """Results of an A/B test comparing two agent configurations."""

    agent_a_name: str
    agent_b_name: str
    dataset_name: str
    total_tasks: int

    # Per-dimension comparison
    dimensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Overall metrics
    overall: dict[str, Any] = field(default_factory=dict)
    # Statistical significance
    significance: dict[str, Any] = field(default_factory=dict)
    # Individual task details
    task_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_a": self.agent_a_name,
            "agent_b": self.agent_b_name,
            "dataset": self.dataset_name,
            "total_tasks": self.total_tasks,
            "dimensions": self.dimensions,
            "overall": self.overall,
            "significance": self.significance,
            "task_details": self.task_details[:20],  # cap for readability
        }


class ABTestRunner:
    """Run A/B comparison tests between two agent configurations."""

    def __init__(self, task_runner: TaskRunner | None = None) -> None:
        self._runner = task_runner or TaskRunner()

    def compare(
        self,
        dataset: TaskDataset,
        agent_a: dict[str, Any],
        agent_b: dict[str, Any],
        *,
        label_a: str | None = None,
        label_b: str | None = None,
        **kwargs: Any,
    ) -> ABTestSummary:
        """Compare two agent configurations on the same dataset.

        Args:
            dataset: Tasks to evaluate on.
            agent_a: Dict with keys like ``agent_type``, ``model``, ``temperature``.
            agent_b: Same schema as agent_a.
            label_a/b: Display names for the two agents.
            **kwargs: Forwarded to :meth:`TaskRunner.run_batch`.
        """
        name_a = label_a or self._make_label(agent_a, "A")
        name_b = label_b or self._make_label(agent_b, "B")

        # Run agent A
        print(f"\n=== Running Agent A: {name_a} ===")
        outcomes_a, summary_a = self._runner.run_batch(
            dataset,
            agent_type=agent_a.get("agent_type"),
            model=agent_a.get("model"),
            temperature=agent_a.get("temperature"),
            max_steps=agent_a.get("max_steps"),
            agent_name_prefix=f"{name_a}-",
            **kwargs,
        )

        # Run agent B
        print(f"\n=== Running Agent B: {name_b} ===")
        outcomes_b, summary_b = self._runner.run_batch(
            dataset,
            agent_type=agent_b.get("agent_type"),
            model=agent_b.get("model"),
            temperature=agent_b.get("temperature"),
            max_steps=agent_b.get("max_steps"),
            agent_name_prefix=f"{name_b}-",
            **kwargs,
        )

        return self._build_summary(
            name_a, name_b, dataset.name, outcomes_a, outcomes_b, summary_a, summary_b
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_label(agent_cfg: dict[str, Any], default: str) -> str:
        model = agent_cfg.get("model", "")
        agent_type = agent_cfg.get("agent_type", "react")
        parts = [agent_type]
        if model:
            parts.append(str(model).split("/")[-1])
        return "-".join(parts)

    def _build_summary(
        self,
        name_a: str,
        name_b: str,
        dataset_name: str,
        outcomes_a: list[RunOutcome],
        outcomes_b: list[RunOutcome],
        summary_a: BatchSummary | None,
        summary_b: BatchSummary | None,
    ) -> ABTestSummary:
        total = len(outcomes_a)

        # Overall metrics comparison
        overall = self._compare_overall(summary_a, summary_b)

        # Per-dimension comparison
        dimensions = self._compare_dimensions(summary_a, summary_b)

        # Statistical significance (paired t-test on per-task scores)
        significance = self._compute_significance(outcomes_a, outcomes_b)

        # Per-task details
        task_details = self._build_task_details(outcomes_a, outcomes_b)

        return ABTestSummary(
            agent_a_name=name_a,
            agent_b_name=name_b,
            dataset_name=dataset_name,
            total_tasks=total,
            dimensions=dimensions,
            overall=overall,
            significance=significance,
            task_details=task_details,
        )

    @staticmethod
    def _compare_overall(
        summary_a: BatchSummary | None,
        summary_b: BatchSummary | None,
    ) -> dict[str, Any]:
        if not summary_a or not summary_b:
            return {}

        def _row(label: str, s: BatchSummary) -> dict[str, Any]:
            return {
                "agent": label,
                "success_rate": round(s.overall_success_rate, 4),
                "quality_score": round(s.overall_quality_score, 4),
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "total_tokens": s.total_tokens,
                "total_cost": round(s.total_cost, 6),
            }

        row_a = _row("A", summary_a)
        row_b = _row("B", summary_b)

        return {
            "agent_a": row_a,
            "agent_b": row_b,
            "delta": {
                "success_rate": round(row_a["success_rate"] - row_b["success_rate"], 4),
                "quality_score": round(row_a["quality_score"] - row_b["quality_score"], 4),
                "avg_latency_ratio": round(
                    row_a["avg_latency_ms"] / max(row_b["avg_latency_ms"], 1), 4
                ),
                "total_tokens": row_a["total_tokens"] - row_b["total_tokens"],
                "total_cost": round(row_a["total_cost"] - row_b["total_cost"], 6),
            },
        }

    @staticmethod
    def _compare_dimensions(
        summary_a: BatchSummary | None,
        summary_b: BatchSummary | None,
    ) -> dict[str, Any]:
        if not summary_a or not summary_b:
            return {}
        result: dict[str, Any] = {}
        all_dims = set(list(summary_a.dimension_summaries.keys()) + list(summary_b.dimension_summaries.keys()))
        for dim in sorted(all_dims):
            ds_a = summary_a.dimension_summaries.get(dim)
            ds_b = summary_b.dimension_summaries.get(dim)
            if ds_a and ds_b:
                result[dim] = {
                    "agent_a": {
                        "mean": round(ds_a.mean_score, 4),
                        "pass_rate": round(ds_a.pass_rate, 4),
                    },
                    "agent_b": {
                        "mean": round(ds_b.mean_score, 4),
                        "pass_rate": round(ds_b.pass_rate, 4),
                    },
                    "mean_delta": round(ds_a.mean_score - ds_b.mean_score, 4),
                    "pass_rate_delta": round(ds_a.pass_rate - ds_b.pass_rate, 4),
                }
        return result

    @staticmethod
    def _compute_significance(
        outcomes_a: list[RunOutcome],
        outcomes_b: list[RunOutcome],
    ) -> dict[str, Any]:
        """Paired comparison on latency and token metrics."""
        if len(outcomes_a) < 3 or len(outcomes_a) != len(outcomes_b):
            return {"note": "need >= 3 paired samples for significance test"}

        latencies_a = [o.run.total_latency_ms for o in outcomes_a]
        latencies_b = [o.run.total_latency_ms for o in outcomes_b]
        tokens_a = [o.run.tokens.total_tokens for o in outcomes_a]
        tokens_b = [o.run.tokens.total_tokens for o in outcomes_b]

        def _paired_ttest(a: list[float], b: list[float]) -> dict[str, Any]:
            """Simple paired t-test implementation."""
            n = len(a)
            diffs = [ai - bi for ai, bi in zip(a, b, strict=False)]
            mean_diff = statistics.mean(diffs)
            if n <= 1:
                return {"mean_diff": mean_diff, "significant": False}
            # Standard error of the difference
            variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
            se = (variance / n) ** 0.5 if variance > 0 else 0.0
            t_stat = mean_diff / se if se > 0 else 0.0
            # Approximate p-value using normal approximation (two-tailed)
            # For |t| > 2 we call it "significant" at ~5% level
            significant = abs(t_stat) > 2.0
            return {
                "mean_diff": round(mean_diff, 2),
                "t_statistic": round(t_stat, 4),
                "significant": significant,
                "n": n,
            }

        return {
            "latency": _paired_ttest(latencies_a, latencies_b),
            "tokens": _paired_ttest(tokens_a, tokens_b),
        }

    @staticmethod
    def _build_task_details(
        outcomes_a: list[RunOutcome],
        outcomes_b: list[RunOutcome],
    ) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        for oa, ob in zip(outcomes_a, outcomes_b, strict=False):
            details.append({
                "task_id": oa.task.task_id,
                "input_preview": oa.task.input[:80],
                "agent_a": {
                    "status": oa.run.status.value,
                    "latency_ms": oa.run.total_latency_ms,
                    "tokens": oa.run.tokens.total_tokens,
                    "output_preview": (oa.run.final_output or "")[:80],
                },
                "agent_b": {
                    "status": ob.run.status.value,
                    "latency_ms": ob.run.total_latency_ms,
                    "tokens": ob.run.tokens.total_tokens,
                    "output_preview": (ob.run.final_output or "")[:80],
                },
                "latency_delta_ms": oa.run.total_latency_ms - ob.run.total_latency_ms,
                "token_delta": oa.run.tokens.total_tokens - ob.run.tokens.total_tokens,
            })
        return details
