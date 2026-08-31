"""Regression baseline: save evaluation results as a baseline and compare future runs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_eval.config import load_config
from agent_eval.evaluation.engine import EvaluationEngine
from agent_eval.logger import setup_logger
from agent_eval.trace.storage import JSONLStorage

logger = setup_logger("agent_eval.evaluation.baseline")


def _baselines_dir() -> Path:
    """Return the directory where baseline JSON files live (outputs/baselines/)."""
    cfg = load_config()
    d = Path(cfg.storage.output_dir) / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_baseline_id() -> str:
    # Use millisecond timestamp + short uuid suffix to avoid collisions even within the same second
    return f"baseline_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ============================================================
# Data model
# ============================================================


@dataclass
class BaselineMeta:
    """Saved baseline record: metadata + serialized BatchSummary."""

    baseline_id: str
    name: str
    created_at: str
    run_ids: list[str]
    dataset_id: str | None
    agent_name: str | None
    summary: dict[str, Any]  # serialized BatchSummary from BatchSummary.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "name": self.name,
            "created_at": self.created_at,
            "run_ids": self.run_ids,
            "dataset_id": self.dataset_id,
            "agent_name": self.agent_name,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineMeta:
        return cls(
            baseline_id=data["baseline_id"],
            name=data["name"],
            created_at=data["created_at"],
            run_ids=data["run_ids"],
            dataset_id=data.get("dataset_id"),
            agent_name=data.get("agent_name"),
            summary=data["summary"],
        )


# ============================================================
# Persistence helpers
# ============================================================


def _baseline_path(baseline_id: str) -> Path:
    return _baselines_dir() / f"{baseline_id}.json"


def _write_baseline(meta: BaselineMeta) -> Path:
    path = _baseline_path(meta.baseline_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"Baseline saved: {meta.baseline_id} -> {path}")
    return path


def _read_baseline(baseline_id: str) -> BaselineMeta:
    path = _baseline_path(baseline_id)
    if not path.exists():
        raise FileNotFoundError(f"Baseline '{baseline_id}' not found at {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return BaselineMeta.from_dict(data)


# ============================================================
# Public API
# ============================================================


def save_baseline(
    engine: EvaluationEngine | None = None,
    run_ids: list[str] | None = None,
    name: str = "",
    dataset_id: str | None = None,
    agent_name: str | None = None,
    storage: JSONLStorage | None = None,
) -> str:
    """Evaluate runs and store the BatchSummary + metadata as a baseline.

    Returns the ``baseline_id`` of the newly saved baseline.
    """
    storage = storage or JSONLStorage()
    engine = engine or EvaluationEngine(storage=storage)

    # Resolve run_ids
    if run_ids is None:
        runs = storage.list_runs()
        run_ids = [r.run_id for r in runs]

    if not run_ids:
        raise ValueError("No runs available to create a baseline.")

    # Evaluate all runs (save_summary=False to avoid writing a stray batch_summary file)
    _per_run, summary = engine.evaluate_runs(run_ids, save_summary=False)

    baseline_id = _make_baseline_id()
    meta = BaselineMeta(
        baseline_id=baseline_id,
        name=name or baseline_id,
        created_at=_utc_now_iso(),
        run_ids=list(run_ids),
        dataset_id=dataset_id,
        agent_name=agent_name,
        summary=summary.to_dict(),
    )
    _write_baseline(meta)
    return baseline_id


def load_baseline(baseline_id: str) -> BaselineMeta:
    """Load a previously saved baseline JSON."""
    return _read_baseline(baseline_id)


def list_baselines() -> list[BaselineMeta]:
    """List all saved baselines, newest first."""
    d = _baselines_dir()
    results: list[BaselineMeta] = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            results.append(BaselineMeta.from_dict(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping malformed baseline file {p}: {e}")
    # Secondary sort by created_at desc for safety
    results.sort(key=lambda b: b.created_at, reverse=True)
    return results


def delete_baseline(baseline_id: str) -> bool:
    """Delete a saved baseline file. Returns True if deleted, False if not found."""
    path = _baseline_path(baseline_id)
    if path.exists():
        path.unlink()
        logger.info(f"Baseline deleted: {baseline_id}")
        return True
    return False


# ============================================================
# Comparison
# ============================================================


# Map BatchSummary top-level keys that are comparable scalar rates.
# Each is (label_in_note, json_key_in_summary).
_COMPARABLE_OVERALL_KEYS = [
    ("overall_success_rate", "Overall success rate"),
    ("overall_quality_score", "Overall quality score"),
]


def compare_to_baseline(
    baseline_id: str,
    current_run_ids: list[str] | None = None,
    engine: EvaluationEngine | None = None,
    storage: JSONLStorage | None = None,
) -> dict[str, Any]:
    """Evaluate current runs and compare against a saved baseline.

    Returns a dict with:
        overall_delta: float              # current_success_rate - baseline_success_rate
        regressions: list[str]            # dimension names that got WORSE
        improvements: list[str]           # dimension names that improved
        dimension_deltas: dict[str, float]  # {dimension_name: delta}
        overall_metrics_delta: dict[str, float]  # top-level metric deltas
        note: str                         # human-readable verdict
        baseline_id: str
        baseline_name: str
        current_run_count: int
    """
    storage = storage or JSONLStorage()
    engine = engine or EvaluationEngine(storage=storage)

    baseline = _read_baseline(baseline_id)
    baseline_summary = baseline.summary

    # Resolve current runs
    if current_run_ids is None:
        runs = storage.list_runs()
        current_run_ids = [r.run_id for r in runs]

    if not current_run_ids:
        return {
            "overall_delta": 0.0,
            "regressions": [],
            "improvements": [],
            "dimension_deltas": {},
            "overall_metrics_delta": {},
            "note": "No current runs to compare.",
            "baseline_id": baseline_id,
            "baseline_name": baseline.name,
            "current_run_count": 0,
        }

    # Evaluate current runs
    _per_run, current_summary = engine.evaluate_runs(current_run_ids, save_summary=False)
    current_dict = current_summary.to_dict()

    # --- Dimension-level comparison via "pass_rate" ---
    baseline_dims = baseline_summary.get("dimensions", {})
    current_dims = current_dict.get("dimensions", {})
    all_dim_names = set(baseline_dims.keys()) | set(current_dims.keys())

    dimension_deltas: dict[str, float] = {}
    regressions: list[str] = []
    improvements: list[str] = []

    for dim_name in all_dim_names:
        b_rate = baseline_dims.get(dim_name, {}).get("pass_rate")
        c_rate = current_dims.get(dim_name, {}).get("pass_rate")
        if b_rate is None or c_rate is None:
            continue
        delta = round(c_rate - b_rate, 4)
        dimension_deltas[dim_name] = delta
        if delta < -1e-6:
            regressions.append(dim_name)
        elif delta > 1e-6:
            improvements.append(dim_name)

    # --- Overall top-level metric deltas ---
    overall_metrics_delta: dict[str, float] = {}
    for key, _label in _COMPARABLE_OVERALL_KEYS:
        b_val = baseline_summary.get(key)
        c_val = current_dict.get(key)
        if b_val is None or c_val is None:
            continue
        overall_metrics_delta[key] = round(c_val - b_val, 4)

    overall_delta = overall_metrics_delta.get("overall_success_rate", 0.0)

    # --- Build note ---
    note = _build_comparison_note(
        overall_delta, regressions, improvements, dimension_deltas, overall_metrics_delta
    )

    return {
        "overall_delta": round(overall_delta, 4),
        "regressions": regressions,
        "improvements": improvements,
        "dimension_deltas": dimension_deltas,
        "overall_metrics_delta": overall_metrics_delta,
        "note": note,
        "baseline_id": baseline_id,
        "baseline_name": baseline.name,
        "current_run_count": len(current_run_ids),
        "baseline_run_count": len(baseline.run_ids),
    }


def _build_comparison_note(
    overall_delta: float,
    regressions: list[str],
    improvements: list[str],
    dimension_deltas: dict[str, float],
    overall_metrics_delta: dict[str, float],
) -> str:
    """Produce a concise human-readable verdict string."""
    lines: list[str] = []

    if not regressions and not improvements:
        return "No regression detected — all metrics unchanged."

    if regressions:
        worst = min(regressions, key=lambda d: dimension_deltas.get(d, 0.0))
        pct_drop = abs(dimension_deltas.get(worst, 0.0)) * 100
        lines.append(
            f"⚠ Regression detected: '{worst}' regressed by {pct_drop:.1f}%"
        )
        if overall_delta < -1e-6:
            lines.append(
                f"  Overall success rate delta: {overall_delta * 100:+.1f} pp"
            )

    if improvements:
        best = max(improvements, key=lambda d: dimension_deltas.get(d, 0.0))
        pct_gain = dimension_deltas.get(best, 0.0) * 100
        lines.append(
            f"✓ Improvement: '{best}' improved by {pct_gain:.1f}%"
        )

    return " | ".join(lines) if lines else "No regression detected."
