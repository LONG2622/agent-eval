# 8.13 Phase 3 - FastAPI Server
"""REST API endpoints for the agent-eval server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from agent_eval import __version__
from agent_eval.config import load_config
from agent_eval.evaluation import ABTestRunner, LLMJudgeEvaluator
from agent_eval.task import TaskDataset, TaskItem, TaskRunner
from agent_eval.trace import AnnotationRecord, RunStatus, SpanType

from agent_eval.server.models import AnnotationRequest, EvalBatchRequest, RunTaskRequest
from agent_eval.server.state import _run_to_dict, _span_to_dict, _storage, get_engine

router = APIRouter()


# ============================================================
# Health & Version
# ============================================================


@router.get("/api/health")
def health():
    cfg = load_config()
    runs = _storage.list_runs()
    return {
        "status": "ok",
        "version": __version__,
        "total_runs": len(runs),
        "storage_backend": cfg.storage.backend,
        "default_model": cfg.llm.default_model,
    }


@router.get("/api/config")
def get_config():
    cfg = load_config()
    return JSONResponse(cfg.model_dump())


@router.get("/api/models")
def list_models():
    """Return the list of selectable models with their UI metadata."""
    from agent_eval.config import list_model_profiles

    cfg = load_config()
    profiles = list_model_profiles()
    models = []
    for p in profiles:
        models.append({
            "id": p.id,
            "display_name": p.display_name,
            "model": p.model,
            "provider": p.provider,
            "description": p.description,
            "supports_function_calling": p.supports_function_calling,
            "supports_chinese": p.supports_chinese,
            "is_default": p.id == cfg.llm.default_model or p.model == cfg.llm.default_model,
        })
    return {
        "default_model": cfg.llm.default_model,
        "models": models,
        "total": len(models),
    }


# ============================================================
# Runs
# ============================================================


@router.get("/api/runs")
def list_runs(
    limit: int = Query(50, ge=1, le=500),
    task_id: Optional[str] = None,
    status: Optional[str] = None,
):
    runs = _storage.list_runs(task_id=task_id)
    if status:
        runs = [r for r in runs if r.status.value == status]
    results = []
    for r in runs[:limit]:
        results.append({
            "run_id": r.run_id,
            "task_id": r.task_id,
            "agent_name": r.agent_name,
            "status": r.status.value,
            "input_text": r.input_text[:100],
            "final_output": (r.final_output or "")[:100],
            "total_latency_ms": r.total_latency_ms,
            "total_steps": r.total_steps,
            "tokens": r.tokens.model_dump(mode="json"),
            "total_cost": r.total_cost,
            "started_at": r.started_at,
        })
    return {"total": len(runs), "runs": results}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _storage.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    spans = _storage.load_spans(run_id)
    engine = get_engine()
    eval_results = engine.get_run_results(run_id)

    return {
        "run": _run_to_dict(run),
        "spans": [_span_to_dict(s) for s in spans],
        "evaluation": [r.to_storage_dict() for r in (eval_results or [])],
    }


@router.post("/api/runs")
def create_run(req: RunTaskRequest):
    runner = TaskRunner(storage=_storage)
    task_item = TaskItem(input=req.task, expected_output=req.expected_output)
    if req.task_id:
        task_item.task_id = req.task_id

    outcome = runner.run_single(
        task_item,
        agent_type=req.agent_type,
        model=req.model,
        temperature=req.temperature,
        max_steps=req.max_steps,
        auto_evaluate=True,
    )

    engine = get_engine()
    eval_results = engine.get_run_results(outcome.run.run_id)

    return {
        "run": _run_to_dict(outcome.run),
        "spans": [_span_to_dict(s) for s in runner.storage.load_spans(outcome.run.run_id)],
        "evaluation": [r.to_storage_dict() for r in (eval_results or [])],
    }


@router.post("/api/runs/{run_id}/evaluate")
def evaluate_run(run_id: str):
    engine = get_engine()
    try:
        results = engine.evaluate_run(run_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {
        "run_id": run_id,
        "results_count": len(results),
        "results": [r.to_storage_dict() for r in results],
    }


# ============================================================
# Batch Evaluation
# ============================================================


@router.post("/api/eval")
def eval_batch(req: EvalBatchRequest):
    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

    tasks = TaskDataset.from_jsonl(dataset_path)
    if req.sample > 0:
        tasks = tasks.sample(req.sample)
    if not tasks.items:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    runner = TaskRunner(storage=_storage)
    outcomes, summary = runner.run_batch(
        tasks,
        agent_type=req.agent_type,
        model=req.model,
        temperature=req.temperature,
        max_steps=req.max_steps,
        workers=req.workers,
        max_retries=req.retries,
    )

    return {
        "total_tasks": len(tasks.items),
        "success_count": sum(1 for o in outcomes if o.run.status == RunStatus.SUCCESS),
        "fail_count": sum(1 for o in outcomes if o.run.status != RunStatus.SUCCESS),
        "summary": summary.to_dict() if summary else None,
        "run_ids": [o.run.run_id for o in outcomes],
    }


# ============================================================
# A/B Compare
# ============================================================


@router.post("/api/compare")
def compare(req: EvalBatchRequest, model_b: Optional[str] = None, agent_b: str = "react"):
    dataset_path = Path(req.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_path}")

    tasks = TaskDataset.from_jsonl(dataset_path)
    if req.sample > 0:
        tasks = tasks.sample(req.sample)
    if not tasks.items:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    runner = ABTestRunner()
    summary = runner.compare(
        tasks,
        agent_a={"agent_type": req.agent_type, "model": req.model},
        agent_b={"agent_type": agent_b, "model": model_b},
        label_a=f"A-{req.model or req.agent_type}",
        label_b=f"B-{model_b or agent_b}",
    )
    return summary.to_dict()


# ============================================================
# LLM-as-Judge
# ============================================================


@router.post("/api/judge/{run_id}")
def judge_run(run_id: str, judge_model: Optional[str] = None):
    run = _storage.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    spans = _storage.load_spans(run_id)
    judge = LLMJudgeEvaluator(judge_model=judge_model)
    results = judge.evaluate(run, spans)
    # Persist
    engine = get_engine()
    engine._persist_run_results(run_id, results)
    return {
        "run_id": run_id,
        "results_count": len(results),
        "results": [r.to_storage_dict() for r in results],
    }


# ============================================================
# Trace Replay
# ============================================================


@router.get("/api/runs/{run_id}/trace")
def get_run_trace(run_id: str):
    """Get structured trace data for replay visualization."""
    run = _storage.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    spans = _storage.load_spans(run_id)
    engine = get_engine()
    eval_results = engine.get_run_results(run_id)

    # Group spans by step_index for structured display
    steps: dict[int, list[dict[str, Any]]] = {}
    for span in spans:
        step = span.step_index
        if step not in steps:
            steps[step] = []
        steps[step].append(_span_to_dict(span))

    # Build step summaries
    step_summaries = []
    for step_idx in sorted(steps.keys()):
        step_spans = steps[step_idx]
        thoughts = [s for s in step_spans if s["span_type"] == SpanType.THOUGHT.value]
        llm_calls = [s for s in step_spans if s["span_type"] == SpanType.LLM_CALL.value]
        tool_calls = [s for s in step_spans if s["span_type"] == SpanType.TOOL_CALL.value]
        agent_steps = [s for s in step_spans if s["span_type"] == SpanType.AGENT_STEP.value]

        step_summaries.append({
            "step_index": step_idx,
            "thoughts": thoughts,
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "agent_steps": agent_steps,
            "total_spans": len(step_spans),
            "has_error": any(not s.get("is_success", True) for s in step_spans),
            "latency_ms": sum(s.get("latency_ms", 0) for s in step_spans),
        })

    return {
        "run": _run_to_dict(run),
        "steps": step_summaries,
        "total_steps": len(step_summaries),
        "evaluation": [r.to_storage_dict() for r in (eval_results or [])],
    }


# ============================================================
# Annotations
# ============================================================


@router.get("/api/runs/{run_id}/annotations")
def get_run_annotations(run_id: str):
    annotations = _storage.load_annotations(run_id=run_id)
    return {"run_id": run_id, "annotations": [a.to_storage_dict() for a in annotations]}


@router.post("/api/runs/{run_id}/annotate")
def create_annotation(run_id: str, req: AnnotationRequest):
    run = _storage.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    annotation = AnnotationRecord(
        run_id=run_id,
        annotator=req.annotator,
        score=req.score,
        labels=req.labels,
        comment=req.comment,
    )
    _storage.save_annotation(annotation)
    return {"annotation": annotation.to_storage_dict()}


@router.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: str):
    deleted = _storage.delete_annotation(annotation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")
    return {"success": True, "annotation_id": annotation_id}


@router.get("/api/annotations")
def list_annotations(limit: int = Query(100, ge=1, le=500)):
    annotations = _storage.load_annotations()
    annotations.sort(key=lambda a: a.created_at, reverse=True)
    return {"total": len(annotations), "annotations": [a.to_storage_dict() for a in annotations[:limit]]}


# ============================================================
# Dashboard Stats
# ============================================================


@router.get("/api/dashboard/stats")
def dashboard_stats():
    runs = _storage.list_runs()
    engine = get_engine()

    # Compute basic stats
    total = len(runs)
    success = sum(1 for r in runs if r.status == RunStatus.SUCCESS)
    failed = sum(1 for r in runs if r.status == RunStatus.FAILED)
    total_tokens = sum(r.tokens.total_tokens for r in runs)
    total_cost = sum(r.total_cost for r in runs)
    avg_latency = (sum(r.total_latency_ms for r in runs) / total) if total else 0

    # Last 7 days runs
    from collections import Counter
    # Extract runs by date
    by_status = Counter(r.status.value for r in runs)

    # Aggregate evaluation scores if available
    dim_scores: dict[str, list[float]] = {}
    for r in runs[:20]:  # Use last 20 runs for dimension stats
        results = engine.get_run_results(r.run_id) or []
        for er in results:
            if er.sub_metric is None:
                dim_scores.setdefault(er.dimension.value, []).append(er.score)

    return {
        "total_runs": total,
        "success_count": success,
        "failed_count": failed,
        "success_rate": round(success / max(total, 1), 4),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6),
        "avg_latency_ms": round(avg_latency, 1),
        "by_status": dict(by_status),
        "dimension_scores": {
            k: {
                "mean": round(sum(v) / len(v), 4) if v else 0,
                "count": len(v),
            }
            for k, v in dim_scores.items()
        },
    }


# ============================================================
# Error Classification API
# ============================================================


@router.get("/api/errors/summary")
def get_error_summary():
    """Get classified error summary for all failed runs."""
    from agent_eval.evaluation.error_classifier import classify_all_runs

    summary = classify_all_runs(_storage, limit=20)
    return {
        "total_runs": summary.total_runs,
        "total_failed": summary.total_failed,
        "total_success": summary.total_success,
        "failure_rate": round(summary.failure_rate, 4),
        "by_category": list(summary.by_category.values()),
        "recent_errors": [
            {
                "run_id": e.run_id,
                "task": e.task[:120],
                "agent_name": e.agent_name,
                "category_code": e.category.code,
                "category_label": e.category.label,
                "error_message": e.error_message[:300],
                "latency_ms": e.latency_ms,
                "steps": e.steps,
            }
            for e in summary.recent_errors
        ],
    }


# ============================================================
# Annotation vs Auto-Evaluation Comparison API
# ============================================================


@router.get("/api/comparison/report")
def get_comparison_report():
    """Get comparison report between human annotations and auto-evaluation."""
    from agent_eval.report import run_comparison

    summary, items = run_comparison(_storage)
    return {
        "summary": summary.to_dict(),
        "items": [item.to_dict() for item in items],
    }
