# 8.13 Phase 3 - FastAPI Server
"""FastAPI server exposing agent-eval capabilities via REST API + Web Dashboard."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_eval import __version__
from agent_eval.config import load_config, reset_config
from agent_eval.evaluation import ABTestRunner, EvaluationEngine, LLMJudgeEvaluator
from agent_eval.task import TaskDataset, TaskItem, TaskRunner
from agent_eval.trace import AnnotationRecord, JSONLStorage, RunStatus, SpanType

# ============================================================
# App setup
# ============================================================

# Force-reload config from .env to avoid stale cached values
reset_config()
_cfg = load_config(force_reload=True)

app = FastAPI(
    title="Agent Eval Server",
    version=__version__,
    description="REST API + Web Dashboard for agent-eval.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Dependencies
# ============================================================

_storage = JSONLStorage()
_engine: EvaluationEngine | None = None


def get_engine() -> EvaluationEngine:
    global _engine
    if _engine is None:
        _engine = EvaluationEngine(storage=_storage)
    return _engine


def _run_to_dict(run) -> dict[str, Any]:
    """Convert a RunRecord to a JSON-serializable dict."""
    d = run.to_storage_dict()
    return d


def _span_to_dict(span) -> dict[str, Any]:
    return span.to_storage_dict()


# ============================================================
# Pydantic request models
# ============================================================


class RunTaskRequest(BaseModel):
    task: str
    agent_type: str = "react"
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_steps: int = 10
    expected_output: Optional[str] = None
    task_id: Optional[str] = None


class EvalBatchRequest(BaseModel):
    dataset_path: str
    agent_type: str = "react"
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_steps: int = 10
    sample: int = 0
    workers: int = 1
    retries: int = 2


class AnnotationRequest(BaseModel):
    score: int = 5
    labels: list[str] = []
    comment: str = ""
    annotator: str = "anonymous"


# ============================================================
# Health & Version
# ============================================================


@app.get("/api/health")
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


@app.get("/api/config")
def get_config():
    cfg = load_config()
    return JSONResponse(cfg.model_dump())


@app.get("/api/models")
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


@app.get("/api/runs")
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


@app.get("/api/runs/{run_id}")
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


@app.post("/api/runs")
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


@app.post("/api/runs/{run_id}/evaluate")
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


@app.post("/api/eval")
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


@app.post("/api/compare")
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


@app.post("/api/judge/{run_id}")
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


@app.get("/api/runs/{run_id}/trace")
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


@app.get("/api/runs/{run_id}/annotations")
def get_run_annotations(run_id: str):
    annotations = _storage.load_annotations(run_id=run_id)
    return {"run_id": run_id, "annotations": [a.to_storage_dict() for a in annotations]}


@app.post("/api/runs/{run_id}/annotate")
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


@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: str):
    deleted = _storage.delete_annotation(annotation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")
    return {"success": True, "annotation_id": annotation_id}


@app.get("/api/annotations")
def list_annotations(limit: int = Query(100, ge=1, le=500)):
    annotations = _storage.load_annotations()
    annotations.sort(key=lambda a: a.created_at, reverse=True)
    return {"total": len(annotations), "annotations": [a.to_storage_dict() for a in annotations[:limit]]}


# ============================================================
# Dashboard Stats
# ============================================================


@app.get("/api/dashboard/stats")
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
# HTML Dashboard
# ============================================================

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Eval Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; }
  .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 32px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 22px; font-weight: 600; }
  .header .version { font-size: 12px; opacity: 0.8; }
  .nav-links { display: flex; gap: 6px; }
  .nav-links a { color: white; text-decoration: none; background: rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 6px; font-size: 13px; }
  .nav-links a.active { background: rgba(255,255,255,0.3); font-weight: 600; }
  .nav-links a:hover { background: rgba(255,255,255,0.25); }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .kpi-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); transition: transform 0.2s; }
  .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
  .kpi-card .label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi-card .value { font-size: 28px; font-weight: 700; margin-top: 8px; }
  .kpi-card .value.success { color: #27ae60; }
  .kpi-card .value.failed { color: #e74c3c; }
  .kpi-card .value.info { color: #3498db; }
  .kpi-card .value.warn { color: #f39c12; }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .chart-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .chart-card h3 { font-size: 15px; margin-bottom: 12px; color: #34495e; }
  .chart-container { width: 100%; height: 280px; }
  .runs-table { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; }
  .runs-table h3 { padding: 16px 20px; border-bottom: 1px solid #ecf0f1; font-size: 15px; color: #34495e; }
  table { width: 100%; border-collapse: collapse; }
  th { background: #f8f9fa; text-align: left; padding: 12px 16px; font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
  td { padding: 12px 16px; border-top: 1px solid #f1f2f6; font-size: 13px; }
  tr:hover td { background: #f8f9fa; }
  .status-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .status-success { background: #d5f5e3; color: #1e8449; }
  .status-failed { background: #fadbd8; color: #c0392b; }
  .status-running { background: #d6eaf8; color: #21618c; }
  .status-pending { background: #fdebd0; color: #ca6f1e; }
  .run-id { font-family: monospace; color: #2980b9; cursor: pointer; text-decoration: none; }
  .run-id:hover { text-decoration: underline; }
  .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
  .loading { text-align: center; padding: 40px; color: #95a5a6; }
  .section-title { font-size: 14px; color: #95a5a6; margin-bottom: 12px; }
  @media (max-width: 768px) {
    .charts-grid { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<div class="header">
  <h1>🧪 Agent Eval Dashboard</h1>
  <div class="nav-links">
    <a href="/" class="active">Dashboard</a>
    <a href="/chat">💬 Chat</a>
    <a href="/errors">🔴 Errors</a>
  </div>
</div>
<div class="container">
  <div class="section-title" id="lastUpdated"></div>
  <div class="kpi-grid" id="kpiGrid">
    <div class="loading">Loading metrics...</div>
  </div>
  <div class="charts-grid">
    <div class="chart-card">
      <h3>📊 Status Distribution</h3>
      <div class="chart-container" id="statusChart"></div>
    </div>
    <div class="chart-card">
      <h3>🎯 Evaluation Scores (by Dimension)</h3>
      <div class="chart-container" id="radarChart"></div>
    </div>
  </div>
  <div class="runs-table">
    <h3>📋 Recent Runs</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Agent</th>
            <th>Status</th>
            <th>Task</th>
            <th>Latency (ms)</th>
            <th>Tokens</th>
            <th>Steps</th>
            <th>Time</th>
            <th>Trace</th>
            <th>Annotate</th>
          </tr>
        </thead>
        <tbody id="runsBody">
          <tr><td colspan="8" class="loading">Loading runs...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
const API = window.location.origin;
const statusColors = { success: '#27ae60', failed: '#e74c3c', running: '#3498db', pending: '#f39c12', timeout: '#9b59b6' };
const statusLabels = { success: 'Success', failed: 'Failed', running: 'Running', pending: 'Pending', timeout: 'Timeout' };

async function loadStats() {
  try {
    const [statsRes, runsRes] = await Promise.all([
      fetch(API + '/api/dashboard/stats'),
      fetch(API + '/api/runs?limit=50')
    ]);
    const stats = await statsRes.json();
    const runsData = await runsRes.json();
    document.getElementById('version').textContent = 'v' + (await fetch(API + '/api/health').then(r => r.json())).version;
    document.getElementById('lastUpdated').textContent = 'Last updated: ' + new Date().toLocaleTimeString();
    renderKPI(stats);
    renderStatusChart(stats);
    renderRadarChart(stats);
    renderRuns(runsData.runs);
  } catch(e) {
    console.error(e);
    document.getElementById('kpiGrid').innerHTML = '<div class="empty-state">❌ Failed to load data. Is the server running?</div>';
  }
}

function renderKPI(stats) {
  const sr = (stats.success_rate * 100).toFixed(1) + '%';
  document.getElementById('kpiGrid').innerHTML = `
    <div class="kpi-card">
      <div class="label">Total Runs</div>
      <div class="value info">${stats.total_runs}</div>
    </div>
    <div class="kpi-card">
      <div class="label">Success Rate</div>
      <div class="value success">${sr}</div>
    </div>
    <div class="kpi-card">
      <div class="label">Failed</div>
      <div class="value failed">${stats.failed_count}</div>
    </div>
    <div class="kpi-card">
      <div class="label">Avg Latency</div>
      <div class="value warn">${stats.avg_latency_ms.toFixed(0)} ms</div>
    </div>
    <div class="kpi-card">
      <div class="label">Total Tokens</div>
      <div class="value info">${stats.total_tokens.toLocaleString()}</div>
    </div>
    <div class="kpi-card">
      <div class="label">Total Cost</div>
      <div class="value warn">$${stats.total_cost.toFixed(4)}</div>
    </div>
  `;
}

function renderStatusChart(stats) {
  const chart = echarts.init(document.getElementById('statusChart'));
  const data = Object.entries(stats.by_status || {}).map(([k, v]) => ({
    name: statusLabels[k] || k,
    value: v,
    itemStyle: { color: statusColors[k] || '#95a5a6' }
  }));
  chart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: true,
      label: { show: true, formatter: '{b}\\n{c}' },
      data: data.length > 0 ? data : [{ name: 'No data', value: 1 }]
    }]
  });
}

function renderRadarChart(stats) {
  const chart = echarts.init(document.getElementById('radarChart'));
  const dims = stats.dimension_scores || {};
  const indicators = Object.keys(dims).map(k => ({
    name: k.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase()),
    max: 1.0
  }));
  const values = Object.values(dims).map(v => v.mean || 0);
  chart.setOption({
    tooltip: {},
    radar: { indicator: indicators.length > 0 ? indicators : [{ name: 'Score', max: 1 }] },
    series: [{
      type: 'radar',
      data: [{
        value: values.length > 0 ? values : [0],
        name: 'Score',
        areaStyle: { opacity: 0.3, color: '#667eea' },
        lineStyle: { color: '#667eea' },
        itemStyle: { color: '#667eea' }
      }]
    }]
  });
}

function renderRuns(runs) {
  if (!runs || runs.length === 0) {
    document.getElementById('runsBody').innerHTML = '<tr><td colspan="10" class="empty-state">No runs yet. Run a task with <code>agent run</code> or via API.</td></tr>';
    return;
  }
  document.getElementById('runsBody').innerHTML = runs.map(r => {
    const statusClass = 'status-' + r.status;
    const time = r.started_at ? new Date(r.started_at).toLocaleString() : '-';
    const preview = (r.input_text || '').slice(0, 60) + ((r.input_text || '').length > 60 ? '...' : '');
    return `<tr>
      <td><a class="run-id" href="/trace/${r.run_id}">${r.run_id.slice(0, 12)}...</a></td>
      <td>${r.agent_name || '-'}</td>
      <td><span class="status-badge ${statusClass}">${statusLabels[r.status] || r.status}</span></td>
      <td title="${r.input_text || ''}">${preview}</td>
      <td>${r.total_latency_ms}</td>
      <td>${r.tokens ? r.tokens.total_tokens : 0}</td>
      <td>${r.total_steps}</td>
      <td>${time}</td>
      <td><a href="/trace/${r.run_id}" style="color:#2980b9;text-decoration:none;">🔍 Trace</a></td>
      <td><a href="/annotate/${r.run_id}" style="color:#27ae60;text-decoration:none;">✏️ Annotate</a></td>
    </tr>`;
  }).join('');
}

window.addEventListener('resize', () => {
  document.querySelectorAll('.chart-container').forEach(el => {
    const inst = echarts.getInstanceByDom(el);
    if (inst) inst.resize();
  });
});

loadStats();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


# ============================================================
# WebSocket for progress updates (placeholder)
# ============================================================


class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    async def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                await self.disconnect(ws)


ws_manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Echo for now; future: subscribe to specific job progress
            await ws.send_json({"type": "ack", "data": data})
    except WebSocketDisconnect:
        await ws_manager.disconnect(ws)


# ============================================================
# Trace Replay Page
# ============================================================

_TRACE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace Replay - Agent Eval</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; }
  .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header a { color: white; text-decoration: none; opacity: 0.9; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  .run-summary { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .run-summary h2 { font-size: 16px; margin-bottom: 12px; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .summary-item { padding: 8px 12px; background: #f8f9fa; border-radius: 8px; }
  .summary-item .label { font-size: 11px; color: #7f8c8d; text-transform: uppercase; }
  .summary-item .value { font-size: 18px; font-weight: 600; margin-top: 4px; }
  .summary-item .value.success { color: #27ae60; }
  .summary-item .value.failed { color: #e74c3c; }
  .trace-timeline { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .trace-timeline h3 { font-size: 15px; margin-bottom: 16px; color: #34495e; }
  .step-card { border: 1px solid #ecf0f1; border-radius: 10px; margin-bottom: 12px; overflow: hidden; }
  .step-header { padding: 12px 16px; cursor: pointer; display: flex; align-items: center; justify-content: space-between; background: #f8f9fa; transition: background 0.2s; }
  .step-header:hover { background: #eef1f5; }
  .step-header.error { background: #fadbd8; }
  .step-info { display: flex; align-items: center; gap: 12px; }
  .step-badge { background: #667eea; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .step-badge.error { background: #e74c3c; }
  .step-title { font-size: 14px; font-weight: 500; }
  .step-meta { font-size: 12px; color: #95a5a6; }
  .step-content { display: none; padding: 16px; }
  .step-content.active { display: block; }
  .span-section { margin-bottom: 12px; }
  .span-section:last-child { margin-bottom: 0; }
  .span-type { font-size: 12px; font-weight: 600; text-transform: uppercase; color: #7f8c8d; margin-bottom: 6px; }
  .span-type.thought { color: #3498db; }
  .span-type.llm_call { color: #9b59b6; }
  .span-type.tool_call { color: #f39c12; }
  .span-type.agent_step { color: #1abc9c; }
  .span-item { background: #f8f9fa; border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; font-size: 13px; }
  .span-item .span-name { font-weight: 500; margin-bottom: 4px; }
  .span-item .span-detail { color: #7f8c8d; font-size: 12px; }
  .span-item pre { background: #2c3e50; color: #ecf0f1; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12px; line-height: 1.5; max-height: 300px; }
  .eval-section { margin-top: 20px; background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .eval-section h3 { font-size: 15px; margin-bottom: 12px; color: #34495e; }
  .eval-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
  .eval-item { padding: 10px; background: #f8f9fa; border-radius: 8px; text-align: center; }
  .eval-item .dim { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
  .eval-item .score { font-size: 24px; font-weight: 700; margin-top: 4px; }
  .eval-item .score.high { color: #27ae60; }
  .eval-item .score.medium { color: #f39c12; }
  .eval-item .score.low { color: #e74c3c; }
  .back-link { display: inline-flex; align-items: center; gap: 6px; color: #667eea; text-decoration: none; font-size: 14px; margin-bottom: 16px; }
  .back-link:hover { text-decoration: underline; }
  .loading { text-align: center; padding: 60px; color: #95a5a6; }
  .error-state { text-align: center; padding: 60px; color: #e74c3c; }
  .annotate-btn { display: inline-block; padding: 8px 16px; background: #27ae60; color: white; text-decoration: none; border-radius: 8px; font-size: 13px; margin-top: 12px; }
  .annotate-btn:hover { background: #219a52; }
</style>
</head>
<body>
<div class="header">
  <h1>🔍 Trace Replay</h1>
  <a href="/">← Back to Dashboard</a>
</div>
<div class="container">
  <a class="back-link" href="/">← Dashboard</a>
  <div id="content">
    <div class="loading">Loading trace...</div>
  </div>
</div>

<script>
const API = window.location.origin;
const runId = window.location.pathname.split('/').pop();

const spanTypeLabels = {
  'thought': '💭 Thought',
  'llm_call': '🤖 LLM Call',
  'tool_call': '🔧 Tool Call',
  'agent_step': '⚡ Agent Step'
};

async function loadTrace() {
  try {
    const res = await fetch(API + '/api/runs/' + runId + '/trace');
    if (!res.ok) throw new Error('Not found');
    const data = await res.json();
    renderTrace(data);
  } catch(e) {
    document.getElementById('content').innerHTML = '<div class="error-state">❌ Failed to load trace. Run not found.</div>';
  }
}

function renderTrace(data) {
  const run = data.run;
  const steps = data.steps;
  
  let html = `
    <div class="run-summary">
      <h2>Run Summary</h2>
      <div class="summary-grid">
        <div class="summary-item">
          <div class="label">Status</div>
          <div class="value ${run.status === 'success' ? 'success' : 'failed'}">${run.status}</div>
        </div>
        <div class="summary-item">
          <div class="label">Task</div>
          <div class="value" style="font-size:14px;">${run.input_text.slice(0, 50)}${run.input_text.length > 50 ? '...' : ''}</div>
        </div>
        <div class="summary-item">
          <div class="label">Agent</div>
          <div class="value" style="font-size:14px;">${run.agent_name || '-'}</div>
        </div>
        <div class="summary-item">
          <div class="label">Steps</div>
          <div class="value">${run.total_steps}</div>
        </div>
        <div class="summary-item">
          <div class="label">Latency</div>
          <div class="value">${run.total_latency_ms} ms</div>
        </div>
        <div class="summary-item">
          <div class="label">Tokens</div>
          <div class="value">${run.tokens ? run.tokens.total_tokens : 0}</div>
        </div>
      </div>
      <a class="annotate-btn" href="/annotate/${run.run_id}">✏️ Add Annotation</a>
    </div>
  `;

  // Evaluation section
  if (data.evaluation && data.evaluation.length > 0) {
    const dims = {};
    for (const ev of data.evaluation) {
      if (ev.sub_metric === null || ev.sub_metric === undefined) {
        dims[ev.dimension] = ev.score;
      }
    }
    if (Object.keys(dims).length > 0) {
      html += '<div class="eval-section"><h3>📊 Evaluation Scores</h3><div class="eval-grid">';
      for (const [dim, score] of Object.entries(dims)) {
        const scoreClass = score >= 0.7 ? 'high' : score >= 0.4 ? 'medium' : 'low';
        html += `
          <div class="eval-item">
            <div class="dim">${dim.replace(/_/g, ' ')}</div>
            <div class="score ${scoreClass}">${score.toFixed(2)}</div>
          </div>
        `;
      }
      html += '</div></div>';
    }
  }

  // Trace timeline
  html += '<div class="trace-timeline" style="margin-top:20px;"><h3>🕐 Execution Trace</h3>';
  
  if (steps.length === 0) {
    html += '<div class="loading">No trace data available</div>';
  } else {
    html += '<div id="steps-container">';
    for (const step of steps) {
      const hasError = step.has_error;
      html += `
        <div class="step-card">
          <div class="step-header ${hasError ? 'error' : ''}" onclick="toggleStep(${step.step_index})">
            <div class="step-info">
              <span class="step-badge ${hasError ? 'error' : ''}">Step ${step.step_index}</span>
              <span class="step-title">${step.total_spans} span(s)</span>
            </div>
            <div class="step-meta">${step.latency_ms} ms ${hasError ? '⚠️' : ''} ▼</div>
          </div>
          <div class="step-content" id="step-${step.step_index}">
            ${renderStepContent(step)}
          </div>
        </div>
      `;
    }
    html += '</div>';
  }
  html += '</div>';

  document.getElementById('content').innerHTML = html;
}

function renderStepContent(step) {
  let html = '';
  
  const sections = [
    ['thoughts', '💭 Thoughts'],
    ['llm_calls', '🤖 LLM Calls'],
    ['tool_calls', '🔧 Tool Calls'],
    ['agent_steps', '⚡ Agent Steps']
  ];

  for (const [key, label] of sections) {
    const items = step[key] || [];
    if (items.length > 0) {
      html += `<div class="span-section"><div class="span-type ${key}">${label}</div>`;
      for (const span of items) {
        html += `<div class="span-item">`;
        if (span.name) {
          html += `<div class="span-name">${span.name} ${span.is_success ? '✅' : '❌'}</div>`;
        }
        if (span.input_data && Object.keys(span.input_data).length > 0) {
          html += `<div class="span-detail">Input:</div><pre>${JSON.stringify(span.input_data, null, 2)}</pre>`;
        }
        if (span.output_data && Object.keys(span.output_data).length > 0) {
          html += `<div class="span-detail">Output:</div><pre>${JSON.stringify(span.output_data, null, 2)}</pre>`;
        }
        if (span.tokens && span.tokens.total_tokens > 0) {
          html += `<div class="span-detail">Tokens: ${span.tokens.total_tokens} (prompt: ${span.tokens.prompt_tokens}, completion: ${span.tokens.completion_tokens})</div>`;
        }
        if (span.latency_ms > 0) {
          html += `<div class="span-detail">Latency: ${span.latency_ms}ms</div>`;
        }
        if (span.exception) {
          html += `<div class="span-detail" style="color:#e74c3c;">Error: ${span.exception}</div>`;
        }
        html += `</div>`;
      }
      html += `</div>`;
    }
  }

  return html || '<div style="color:#95a5a6;">No detailed spans</div>';
}

function toggleStep(index) {
  const el = document.getElementById('step-' + index);
  if (el) {
    el.classList.toggle('active');
  }
}

// Load on start
loadTrace();
</script>
</body>
</html>
"""


@app.get("/trace/{run_id}", response_class=HTMLResponse)
def trace_page(run_id: str):
    return HTMLResponse(content=_TRACE_HTML)


# ============================================================
# Annotation Page
# ============================================================

_ANNOTATE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Annotate - Agent Eval</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; }
  .header { background: linear-gradient(135deg, #27ae60 0%, #1abc9c 100%); color: white; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header a { color: white; text-decoration: none; opacity: 0.9; }
  .container { max-width: 800px; margin: 0 auto; padding: 24px; }
  .back-link { display: inline-flex; align-items: center; gap: 6px; color: #27ae60; text-decoration: none; font-size: 14px; margin-bottom: 16px; }
  .back-link:hover { text-decoration: underline; }
  .run-info { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .run-info h2 { font-size: 15px; margin-bottom: 12px; }
  .info-row { display: flex; gap: 12px; margin-bottom: 8px; font-size: 13px; }
  .info-row .label { color: #7f8c8d; min-width: 80px; }
  .info-row .value { flex: 1; }
  .output-box { background: #f8f9fa; border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
  .annotate-form { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .form-group { margin-bottom: 20px; }
  .form-group label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #34495e; }
  .score-options { display: flex; gap: 8px; }
  .score-btn { flex: 1; padding: 12px; border: 2px solid #ecf0f1; border-radius: 8px; background: white; cursor: pointer; font-size: 18px; font-weight: 600; transition: all 0.2s; }
  .score-btn:hover { border-color: #27ae60; }
  .score-btn.active { border-color: #27ae60; background: #d5f5e3; color: #1e8449; }
  .score-btn .score-label { display: block; font-size: 11px; font-weight: 400; margin-top: 2px; }
  .label-options { display: flex; flex-wrap: wrap; gap: 8px; }
  .label-chip { padding: 6px 14px; border: 1px solid #ecf0f1; border-radius: 16px; background: white; cursor: pointer; font-size: 12px; transition: all 0.2s; }
  .label-chip:hover { border-color: #27ae60; }
  .label-chip.active { background: #27ae60; color: white; border-color: #27ae60; }
  textarea { width: 100%; min-height: 100px; padding: 12px; border: 1px solid #ecf0f1; border-radius: 8px; font-size: 13px; font-family: inherit; resize: vertical; }
  textarea:focus { outline: none; border-color: #27ae60; }
  .submit-btn { width: 100%; padding: 14px; background: #27ae60; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
  .submit-btn:hover { background: #219a52; }
  .submit-btn:disabled { background: #bdc3c7; cursor: not-allowed; }
  .existing-annotations { background: white; border-radius: 12px; padding: 20px; margin-top: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .existing-annotations h3 { font-size: 14px; margin-bottom: 12px; }
  .annotation-item { padding: 12px; background: #f8f9fa; border-radius: 8px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: flex-start; }
  .annotation-item .score-display { font-size: 24px; font-weight: 700; color: #27ae60; margin-right: 12px; }
  .annotation-item .annotation-body { flex: 1; }
  .annotation-item .annotation-meta { font-size: 11px; color: #95a5a6; }
  .annotation-item .annotation-comment { font-size: 13px; margin-top: 4px; }
  .annotation-item .delete-btn { background: none; border: none; color: #e74c3c; cursor: pointer; font-size: 16px; padding: 4px 8px; }
  .loading { text-align: center; padding: 60px; color: #95a5a6; }
  .empty-state { text-align: center; padding: 20px; color: #95a5a6; font-size: 13px; }
  .toast { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); padding: 12px 24px; background: #2c3e50; color: white; border-radius: 8px; font-size: 14px; opacity: 0; transition: opacity 0.3s; }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<div class="header">
  <h1>✏️ Human Annotation</h1>
  <a href="/">← Dashboard</a>
</div>
<div class="container">
  <a class="back-link" href="/">← Dashboard</a>
  <div id="content">
    <div class="loading">Loading...</div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const API = window.location.origin;
const runId = window.location.pathname.split('/').pop();
let selectedScore = 5;
let selectedLabels = [];

const LABEL_OPTIONS = ['correct', 'incorrect', 'complete', 'incomplete', 'relevant', 'irrelevant', 'safe', 'unsafe', 'clear', 'unclear'];

async function loadData() {
  try {
    const [runRes, annRes] = await Promise.all([
      fetch(API + '/api/runs/' + runId),
      fetch(API + '/api/runs/' + runId + '/annotations')
    ]);
    if (!runRes.ok) throw new Error('Not found');
    const runData = await runRes.json();
    const annData = await annRes.json();
    renderPage(runData, annData.annotations);
  } catch(e) {
    document.getElementById('content').innerHTML = '<div class="loading">❌ Failed to load run data</div>';
  }
}

function renderPage(runData, existingAnnotations) {
  const run = runData.run;
  const html = `
    <div class="run-info">
      <h2>Run Details</h2>
      <div class="info-row"><span class="label">Run ID:</span><span class="value">${run.run_id}</span></div>
      <div class="info-row"><span class="label">Status:</span><span class="value">${run.status}</span></div>
      <div class="info-row"><span class="label">Task:</span><span class="value">${run.input_text}</span></div>
      <div class="info-row"><span class="label">Output:</span></div>
      <div class="output-box">${run.final_output || 'No output'}</div>
      <div style="margin-top:12px;"><a href="/trace/${run.run_id}" style="color:#27ae60; text-decoration:none;">🔍 View full trace →</a></div>
    </div>

    <div class="annotate-form">
      <h2 style="font-size:16px; margin-bottom:16px;">Add Annotation</h2>
      
      <div class="form-group">
        <label>Score (1-5)</label>
        <div class="score-options" id="scoreOptions">
          <button type="button" class="score-btn" data-score="1">1<span class="score-label">Poor</span></button>
          <button type="button" class="score-btn" data-score="2">2<span class="score-label">Fair</span></button>
          <button type="button" class="score-btn" data-score="3">3<span class="score-label">OK</span></button>
          <button type="button" class="score-btn" data-score="4">4<span class="score-label">Good</span></button>
          <button type="button" class="score-btn active" data-score="5">5<span class="score-label">Excellent</span></button>
        </div>
      </div>

      <div class="form-group">
        <label>Labels (multi-select)</label>
        <div class="label-options" id="labelOptions">
          ${LABEL_OPTIONS.map(l => `<span class="label-chip" data-label="${l}">${l}</span>`).join('')}
        </div>
      </div>

      <div class="form-group">
        <label>Comment</label>
        <textarea id="commentInput" placeholder="Share your thoughts about this agent run..."></textarea>
      </div>

      <button class="submit-btn" onclick="submitAnnotation()">Submit Annotation</button>
    </div>

    <div class="existing-annotations">
      <h3>Existing Annotations (${existingAnnotations.length})</h3>
      ${existingAnnotations.length === 0 ? '<div class="empty-state">No annotations yet. Be the first!</div>' : 
        existingAnnotations.map(a => `
          <div class="annotation-item" id="ann-${a.annotation_id}">
            <div class="score-display">${'⭐'.repeat(a.score)}</div>
            <div class="annotation-body">
              <div class="annotation-meta">${a.annotator} · ${new Date(a.created_at).toLocaleString()}</div>
              ${a.labels.length > 0 ? `<div style="margin-top:4px;"><span style="background:#eef1f5; padding:2px 8px; border-radius:10px; font-size:11px; margin-right:4px;">${a.labels.join(', ')}</span></div>` : ''}
              ${a.comment ? `<div class="annotation-comment">${a.comment}</div>` : ''}
            </div>
            <button class="delete-btn" onclick="deleteAnnotation('${a.annotation_id}')" title="Delete">🗑️</button>
          </div>
        `).join('')
      }
    </div>
  `;

  document.getElementById('content').innerHTML = html;

  // Bind score selection
  document.querySelectorAll('.score-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.score-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedScore = parseInt(btn.dataset.score);
    });
  });

  // Bind label selection
  document.querySelectorAll('.label-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const label = chip.dataset.label;
      if (selectedLabels.includes(label)) {
        selectedLabels = selectedLabels.filter(l => l !== label);
        chip.classList.remove('active');
      } else {
        selectedLabels.push(label);
        chip.classList.add('active');
      }
    });
  });
}

async function submitAnnotation() {
  const comment = document.getElementById('commentInput').value;
  try {
    const res = await fetch(API + '/api/runs/' + runId + '/annotate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        score: selectedScore,
        labels: selectedLabels,
        comment: comment,
        annotator: 'web_user'
      })
    });
    if (!res.ok) throw new Error('Failed to submit');
    showToast('✅ Annotation saved!');
    setTimeout(() => loadData(), 500);
  } catch(e) {
    showToast('❌ Failed to save');
  }
}

async function deleteAnnotation(annId) {
  if (!confirm('Delete this annotation?')) return;
  try {
    const res = await fetch(API + '/api/annotations/' + annId, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete');
    showToast('🗑️ Deleted!');
    setTimeout(() => loadData(), 500);
  } catch(e) {
    showToast('❌ Failed to delete');
  }
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2000);
}

loadData();
</script>
</body>
</html>
"""


@app.get("/annotate/{run_id}", response_class=HTMLResponse)
def annotate_page(run_id: str):
    return HTMLResponse(content=_ANNOTATE_HTML)


# ============================================================
# Interactive Chat Page
# ============================================================

_CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Agent Chat - Agent Eval</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; height: 100vh; display: flex; flex-direction: column; }
  .header { background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%); color: white; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header a { color: white; text-decoration: none; opacity: 0.9; font-size: 14px; margin-left: 16px; }
  .header a:hover { text-decoration: underline; }
  .nav-links { display: flex; gap: 4px; }
  .nav-links a { background: rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 6px; font-size: 13px; }
  .nav-links a.active { background: rgba(255,255,255,0.3); }
  .chat-container { flex: 1; display: flex; overflow: hidden; }
  .chat-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .chat-sidebar { width: 280px; background: white; border-left: 1px solid #ecf0f1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
  .sidebar-title { font-size: 13px; font-weight: 600; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
  .config-group label { display: block; font-size: 12px; font-weight: 500; margin-bottom: 4px; color: #34495e; }
  .config-group input, .config-group select { width: 100%; padding: 8px 10px; border: 1px solid #ecf0f1; border-radius: 6px; font-size: 13px; font-family: inherit; }
  .config-group input:focus, .config-group select:focus { outline: none; border-color: #ff6b6b; }
  .info-tip { font-size: 12px; color: #95a5a6; line-height: 1.5; background: #f8f9fa; padding: 10px; border-radius: 6px; }
  .messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .message { max-width: 85%; display: flex; gap: 12px; }
  .message.user { align-self: flex-end; }
  .message.agent { align-self: flex-start; }
  .avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
  .message.user .avatar { background: #ff6b6b; color: white; }
  .message.agent .avatar { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
  .bubble { padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 14px; word-break: break-word; }
  .message.user .bubble { background: #ff6b6b; color: white; border-bottom-right-radius: 4px; }
  .message.agent .bubble { background: white; color: #2c3e50; border: 1px solid #ecf0f1; border-bottom-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .agent-meta { font-size: 11px; color: #95a5a6; margin-top: 6px; display: flex; gap: 10px; flex-wrap: wrap; }
  .agent-meta span { background: #f8f9fa; padding: 2px 8px; border-radius: 10px; }
  .steps-box { margin-top: 10px; background: #fafbfc; border: 1px solid #ecf0f1; border-radius: 8px; padding: 10px; max-height: 350px; overflow-y: auto; }
  .steps-box-header { font-size: 12px; font-weight: 600; color: #7f8c8d; margin-bottom: 6px; display: flex; justify-content: space-between; cursor: pointer; }
  .step-item { padding: 6px 10px; margin-bottom: 6px; background: white; border-left: 3px solid #667eea; border-radius: 4px; font-size: 12px; }
  .step-item.tool { border-left-color: #f39c12; }
  .step-item.error { border-left-color: #e74c3c; }
  .step-title { font-weight: 500; font-size: 12px; color: #34495e; margin-bottom: 2px; }
  .step-detail { color: #7f8c8d; font-size: 11px; max-height: 100px; overflow-x: auto; }
  .step-detail pre { margin: 4px 0; background: #2c3e50; color: #ecf0f1; padding: 6px 8px; border-radius: 4px; }
  .eval-scores { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .eval-pill { background: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; border: 1px solid #ecf0f1; }
  .eval-pill.high { border-color: #27ae60; color: #1e8449; }
  .eval-pill.medium { border-color: #f39c12; color: #ca6f1e; }
  .eval-pill.low { border-color: #e74c3c; color: #c0392b; }
  .input-area { border-top: 1px solid #ecf0f1; padding: 16px 24px; background: white; display: flex; gap: 12px; align-items: flex-end; }
  .input-wrap { flex: 1; position: relative; }
  .input-wrap textarea { width: 100%; padding: 12px 14px; border: 1px solid #ecf0f1; border-radius: 10px; font-size: 14px; font-family: inherit; resize: none; min-height: 48px; max-height: 200px; line-height: 1.5; }
  .input-wrap textarea:focus { outline: none; border-color: #ff6b6b; }
  .send-btn { padding: 12px 20px; background: #ff6b6b; color: white; border: none; border-radius: 10px; font-size: 14px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
  .send-btn:hover { background: #e55a5a; }
  .send-btn:disabled { background: #bdc3c7; cursor: not-allowed; }
  .typing { display: inline-flex; gap: 4px; padding: 4px 0; }
  .typing span { width: 6px; height: 6px; background: #95a5a6; border-radius: 50%; animation: typing 1.2s infinite; }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes typing { 0%, 60%, 100% { transform: translateY(0); opacity: 0.4; } 30% { transform: translateY(-4px); opacity: 1; } }
  .empty-chat { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #95a5a6; padding: 40px; text-align: center; }
  .empty-chat .icon { font-size: 48px; margin-bottom: 12px; opacity: 0.5; }
  .empty-chat h3 { color: #7f8c8d; margin-bottom: 6px; }
  .empty-chat p { font-size: 13px; max-width: 400px; line-height: 1.6; }
  .sample-prompts { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-top: 20px; max-width: 600px; }
  .sample-prompt { padding: 12px; background: white; border: 1px solid #ecf0f1; border-radius: 10px; cursor: pointer; font-size: 13px; transition: all 0.2s; text-align: left; color: #34495e; }
  .sample-prompt:hover { border-color: #ff6b6b; transform: translateY(-1px); }
  .message-actions { margin-top: 6px; display: flex; gap: 8px; }
  .message-actions a { font-size: 11px; color: #95a5a6; text-decoration: none; }
  .message-actions a:hover { color: #ff6b6b; text-decoration: underline; }
  @media (max-width: 768px) {
    .chat-sidebar { display: none; }
    .message { max-width: 95%; }
  }
</style>
</head>
<body>
<div class="header">
  <h1>💬 Agent Chat</h1>
  <div class="nav-links">
    <a href="/">Dashboard</a>
    <a href="/chat" class="active">Chat</a>
    <a href="/errors">🔴 Errors</a>
  </div>
</div>
<div class="chat-container">
  <div class="chat-main">
    <div class="messages" id="messages">
      <div class="empty-chat" id="emptyChat">
        <div class="icon">🤖</div>
        <h3>Chat with your Agent</h3>
        <p>Ask anything below. The ReAct Agent will use tools (calculator, search, etc.) to reason step-by-step before answering. Try these examples:</p>
        <div class="sample-prompts">
          <button class="sample-prompt" onclick="sendPrompt('Calculate sqrt(144) + 5^2 step by step')">📐 Calculate sqrt(144) + 5²</button>
          <button class="sample-prompt" onclick="sendPrompt('What is 25% of 800? Answer just the number')">🔢 25% of 800</button>
          <button class="sample-prompt" onclick="sendPrompt('What time is it now?')">🕐 What time is it?</button>
          <button class="sample-prompt" onclick="sendPrompt('Tell me about Agent evaluation frameworks')">📚 About Agent Eval</button>
        </div>
      </div>
    </div>
    <div class="input-area">
      <div class="input-wrap">
        <textarea id="messageInput" placeholder="Type your message... (Shift+Enter for newline, Enter to send)" rows="2"></textarea>
      </div>
      <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
  </div>
  <div class="chat-sidebar">
    <div class="sidebar-title">Agent Configuration</div>
    <div class="config-group">
      <label>Agent Type</label>
      <select id="cfgAgent">
        <option value="react">ReAct (default)</option>
      </select>
    </div>
    <div class="config-group">
      <label>Model</label>
      <select id="cfgModel">
        <option value="">Loading models...</option>
      </select>
      <div id="cfgModelHint" class="info-tip" style="margin-top:6px;font-size:12px;"></div>
    </div>
    <div class="config-group">
      <label>Max Steps</label>
      <input type="number" id="cfgSteps" value="10" min="1" max="50">
    </div>
    <div class="config-group">
      <label>Temperature</label>
      <input type="number" id="cfgTemp" value="0.7" step="0.1" min="0" max="2">
    </div>
    <div class="config-group">
      <label>Expected Output (optional)</label>
      <input type="text" id="cfgExpected" placeholder="Ground truth for eval">
    </div>
    <div class="sidebar-title" style="margin-top:10px;">About</div>
    <div class="info-tip">
      Each message creates a run in the dataset, trace, and evaluation records. You can inspect the full trace or add annotations via Dashboard / links below each answer.
    </div>
  </div>
</div>

<script>
const API = window.location.origin;
const messagesEl = document.getElementById('messages');
const emptyChatEl = document.getElementById('emptyChat');
const inputEl = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const cfgModelSel = document.getElementById('cfgModel');
const cfgModelHint = document.getElementById('cfgModelHint');

let chatId = 0;
let MODEL_LIST = [];

// ---- Load available models and populate the dropdown ----
fetch(API + '/api/models')
  .then(r => r.json())
  .then(data => {
    MODEL_LIST = data.models || [];
    cfgModelSel.innerHTML = '';
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = `📌 Default (${data.default_model || 'from config'})`;
    cfgModelSel.appendChild(defaultOpt);
    for (const m of MODEL_LIST) {
      const opt = document.createElement('option');
      opt.value = m.id;
      const badges = [];
      if (m.supports_chinese) badges.push('🇨🇳 中文');
      if (!m.supports_function_calling) badges.push('⚠️ 无FC');
      const badgeStr = badges.length ? ' [' + badges.join(' ') + ']' : '';
      opt.textContent = m.display_name + badgeStr + (m.is_default ? ' ✅' : '');
      if (m.is_default) {
        // Keep the "Default" option selected by default; mark default in UI
        opt.textContent += ' (默认)';
      }
      cfgModelSel.appendChild(opt);
    }
    updateModelHint();
  })
  .catch(err => {
    cfgModelSel.innerHTML = '<option value="">Failed to load models</option>';
  });

cfgModelSel.addEventListener('change', updateModelHint);
function updateModelHint() {
  const id = cfgModelSel.value;
  if (!id) {
    cfgModelHint.textContent = '';
    return;
  }
  const m = MODEL_LIST.find(x => x.id === id);
  if (m && m.description) {
    cfgModelHint.textContent = m.description;
  }
}

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function sendPrompt(p) {
  inputEl.value = p;
  sendMessage();
}

function addUserMessage(text) {
  hideEmpty();
  chatId++;
  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.id = 'msg-' + chatId;
  msg.innerHTML = `
    <div class="avatar">U</div>
    <div class="bubble" style="white-space:pre-wrap;">${escapeHtml(text)}</div>
  `;
  messagesEl.appendChild(msg);
  scrollBottom();
  return chatId;
}

function addAgentPlaceholder() {
  chatId++;
  const msg = document.createElement('div');
  msg.className = 'message agent';
  msg.id = 'msg-' + chatId;
  msg.innerHTML = `
    <div class="avatar">A</div>
    <div class="bubble">
      <div class="typing"><span></span><span></span><span></span></div>
      <div style="font-size:12px;color:#95a5a6;margin-top:8px;">Agent is thinking...</div>
    </div>
  `;
  messagesEl.appendChild(msg);
  scrollBottom();
  return chatId;
}

function fillAgentMessage(id, data) {
  const run = data.run;
  const steps = buildSteps(data.spans);
  const evals = buildEvals(data.evaluation);
  const el = document.getElementById('msg-' + id);
  const statusClass = run.status === 'success' ? 'success' : 'failed';
  const statusIcon = run.status === 'success' ? '✅' : '❌';
  el.querySelector('.bubble').innerHTML = `
    <div style="white-space:pre-wrap;">${escapeHtml(run.final_output || '[no output]')}</div>
    <div class="agent-meta">
      <span>${statusIcon} ${run.status}</span>
      <span>⏱ ${run.total_latency_ms} ms</span>
      <span>🔑 ${run.tokens ? run.tokens.total_tokens : 0} tokens</span>
      <span>⚡ ${run.total_steps} steps</span>
      <span>💰 $${run.total_cost ? run.total_cost.toFixed(5) : '0.00000'}</span>
    </div>
    ${steps}
    ${evals}
    <div class="message-actions">
      <a href="/trace/${run.run_id}" target="_blank">🔍 Full trace</a>
      <a href="/annotate/${run.run_id}" target="_blank">✏️ Annotate</a>
    </div>
  `;
  scrollBottom();
}

function fillAgentError(id, err) {
  const el = document.getElementById('msg-' + id);
  el.querySelector('.bubble').innerHTML = `
    <div style="color:#e74c3c;">❌ Failed to run agent</div>
    <div style="font-size:12px;color:#95a5a6;margin-top:6px;">${escapeHtml(err)}</div>
  `;
}

function buildSteps(spans) {
  if (!spans || spans.length === 0) return '';
  const byStep = {};
  for (const s of spans) {
    const i = s.step_index;
    if (!byStep[i]) byStep[i] = [];
    byStep[i].push(s);
  }
  let html = `<div class="steps-box" onclick="toggleStepsBox(this)">
    <div class="steps-box-header"><span>🕐 Reasoning Steps (${spans.length} spans)</span><span>▼</span></div>
    <div style="display:none;">`;
  const sortedKeys = Object.keys(byStep).map(Number).sort((a,b)=>a-b);
  for (const idx of sortedKeys) {
    const list = byStep[idx];
    const mainSpan = list[0];
    const hasErr = list.some(s => s.is_success === false);
    const isTool = list.some(s => s.span_type === 'tool_call');
    const cls = hasErr ? 'step-item error' : isTool ? 'step-item tool' : 'step-item';
    const label = isTool ? 'Tool Call' : list.some(s=>s.span_type==='llm_call') ? 'LLM Call' : 'Thought/Step';
    const name = mainSpan.name || `Step ${idx}`;
    const outputData = mainSpan.output_data && Object.keys(mainSpan.output_data).length > 0
      ? `<div class="step-detail"><b>Output:</b><pre>${shorten(JSON.stringify(mainSpan.output_data, null, 2), 500)}</pre></div>` : '';
    const inputData = mainSpan.input_data && Object.keys(mainSpan.input_data).length > 0
      ? `<div class="step-detail"><b>Input:</b><pre>${shorten(JSON.stringify(mainSpan.input_data, null, 2), 300)}</pre></div>` : '';
    const tokens = mainSpan.tokens && mainSpan.tokens.total_tokens > 0
      ? `<div class="step-detail">Tokens: ${mainSpan.tokens.total_tokens}</div>` : '';
    const lat = mainSpan.latency_ms ? `<div class="step-detail">Latency: ${mainSpan.latency_ms}ms</div>` : '';
    html += `<div class="${cls}">
      <div class="step-title">${label}: ${escapeHtml(name)} ${hasErr ? '❌' : '✅'}</div>
      ${tokens}${lat}${inputData}${outputData}
    </div>`;
  }
  html += '</div></div>';
  return html;
}

function toggleStepsBox(el) {
  const inner = el.querySelector('div:nth-child(2)');
  const header = el.querySelector('.steps-box-header span:last-child');
  if (inner.style.display === 'none') {
    inner.style.display = 'block';
    header.textContent = '▲';
  } else {
    inner.style.display = 'none';
    header.textContent = '▼';
  }
}

function buildEvals(evals) {
  if (!evals || evals.length === 0) return '';
  const dims = {};
  for (const e of evals) {
    if (e.sub_metric === null || e.sub_metric === undefined) {
      dims[e.dimension] = e.score;
    }
  }
  if (Object.keys(dims).length === 0) return '';
  let html = '<div class="eval-scores">';
  for (const [d, s] of Object.entries(dims)) {
    const cls = s >= 0.7 ? 'high' : s >= 0.4 ? 'medium' : 'low';
    html += `<span class="eval-pill ${cls}">${d.replace(/_/g,' ')}: ${(s*100).toFixed(0)}%</span>`;
  }
  html += '</div>';
  return html;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addUserMessage(text);
  const phId = addAgentPlaceholder();
  sendBtn.disabled = true;
  try {
    const res = await fetch(API + '/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task: text,
        agent_type: document.getElementById('cfgAgent').value,
        model: document.getElementById('cfgModel').value || null,
        temperature: parseFloat(document.getElementById('cfgTemp').value) || null,
        max_steps: parseInt(document.getElementById('cfgSteps').value) || 10,
        expected_output: document.getElementById('cfgExpected').value || null,
      })
    });
    if (!res.ok) {
      const errTxt = await res.text();
      throw new Error(`HTTP ${res.status}: ${errTxt.slice(0, 200)}`);
    }
    const data = await res.json();
    fillAgentMessage(phId, data);
  } catch (e) {
    fillAgentError(phId, String(e));
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideEmpty() {
  if (emptyChatEl) {
    emptyChatEl.remove();
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function shorten(s, n) {
  return s && s.length > n ? s.slice(0, n) + '\n...(truncated)' : s || '';
}
</script>
</body>
</html>
"""


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(content=_CHAT_HTML)


# ============================================================
# Error Classification API + Page
# ============================================================

@app.get("/api/errors/summary")
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


_ERRORS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Error Analysis - Agent Eval</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #2c3e50; }
  .header { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 20px 32px; display: flex; align-items: center; justify-content: space-between; }
  .header h1 { font-size: 22px; font-weight: 600; }
  .nav-links { display: flex; gap: 6px; }
  .nav-links a { color: white; text-decoration: none; background: rgba(255,255,255,0.15); padding: 6px 12px; border-radius: 6px; font-size: 13px; }
  .nav-links a:hover { background: rgba(255,255,255,0.25); }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .kpi-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .kpi-card .label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi-card .value { font-size: 28px; font-weight: 700; margin-top: 8px; }
  .kpi-card .value.red { color: #e74c3c; }
  .kpi-card .value.green { color: #27ae60; }
  .kpi-card .value.orange { color: #f39c12; }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
  .chart-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .chart-card h3 { font-size: 15px; margin-bottom: 12px; color: #34495e; }
  .chart-container { width: 100%; height: 300px; }
  .error-table { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); overflow: hidden; }
  .error-table h3 { padding: 16px 20px; border-bottom: 1px solid #ecf0f1; font-size: 15px; color: #34495e; }
  table { width: 100%; border-collapse: collapse; }
  th { background: #f8f9fa; text-align: left; padding: 10px 16px; font-size: 12px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
  td { padding: 10px 16px; border-top: 1px solid #f1f2f6; font-size: 13px; }
  tr:hover td { background: #f8f9fa; }
  .cat-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; color: white; }
  .cat-llm_timeout { background: #f39c12; }
  .cat-llm_rate_limit { background: #e67e22; }
  .cat-file_system_error { background: #16a085; }
  .cat-llm_auth_error { background: #8e44ad; }
  .cat-llm_bad_request { background: #e74c3c; }
  .cat-llm_not_found { background: #c0392b; }
  .cat-tool_execution_error { background: #d35400; }
  .cat-llm_format_error { background: #9b59b6; }
  .cat-max_steps_exceeded { background: #3498db; }
  .cat-network_error { background: #1abc9c; }
  .cat-internal_error { background: #34495e; }
  .cat-unknown { background: #95a5a6; }
  .run-link { font-family: monospace; color: #2980b9; text-decoration: none; font-size: 12px; }
  .run-link:hover { text-decoration: underline; }
  .err-text { color: #7f8c8d; font-size: 12px; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .loading { text-align: center; padding: 40px; color: #95a5a6; }
  @media (max-width: 768px) {
    .chart-row { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="header">
  <h1>🔴 Error Analysis</h1>
  <div class="nav-links">
    <a href="/">Dashboard</a>
    <a href="/chat">Chat</a>
    <a href="/errors" class="active">Errors</a>
  </div>
</div>
<div class="container">
  <div class="kpi-grid" id="kpiGrid">
    <div class="loading">Loading...</div>
  </div>
  <div class="chart-row">
    <div class="chart-card">
      <h3>Error Distribution</h3>
      <div class="chart-container" id="pieChart"></div>
    </div>
    <div class="chart-card">
      <h3>Success vs Failure</h3>
      <div class="chart-container" id="barChart"></div>
    </div>
  </div>
  <div class="error-table">
    <h3>Recent Failed Runs</h3>
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th>Run ID</th>
          <th>Task</th>
          <th>Error</th>
          <th>Latency</th>
          <th>Steps</th>
        </tr>
      </thead>
      <tbody id="errorTableBody">
        <tr><td colspan="6" class="loading">Loading...</td></tr>
      </tbody>
    </table>
  </div>
</div>
<script>
async function loadData() {
  const res = await fetch('/api/errors/summary');
  const data = await res.json();

  // KPI cards
  document.getElementById('kpiGrid').innerHTML = `
    <div class="kpi-card"><div class="label">Total Runs</div><div class="value">${data.total_runs}</div></div>
    <div class="kpi-card"><div class="label">Success</div><div class="value green">${data.total_success}</div></div>
    <div class="kpi-card"><div class="label">Failed</div><div class="value red">${data.total_failed}</div></div>
    <div class="kpi-card"><div class="label">Failure Rate</div><div class="value orange">${(data.failure_rate * 100).toFixed(1)}%</div></div>
    <div class="kpi-card"><div class="label">Error Types</div><div class="value">${data.by_category.length}</div></div>
  `;

  // Pie chart
  const pieChart = echarts.init(document.getElementById('pieChart'));
  const pieColors = ['#f39c12','#e67e22','#8e44ad','#e74c3c','#c0392b','#d35400','#9b59b6','#3498db','#1abc9c','#34495e','#95a5a6'];
  pieChart.setOption({
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
    legend: { bottom: 0, type: 'scroll' },
    series: [{
      type: 'pie', radius: ['40%','70%'],
      label: { show: true, formatter: '{b}\n{c}', fontSize: 11 },
      data: data.by_category.map((c,i) => ({ value: c.count, name: c.label, itemStyle: { color: pieColors[i % pieColors.length] } })),
    }],
  });

  // Bar chart
  const barChart = echarts.init(document.getElementById('barChart'));
  barChart.setOption({
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ['Success','Failed'] },
    yAxis: { type: 'value' },
    series: [{
      type: 'bar', barWidth: '40%',
      data: [
        { value: data.total_success, itemStyle: { color: '#27ae60' } },
        { value: data.total_failed, itemStyle: { color: '#e74c3c' } },
      ],
      label: { show: true, position: 'top', fontSize: 14, fontWeight: 'bold' },
    }],
  });

  // Error table
  const tbody = document.getElementById('errorTableBody');
  if (data.recent_errors.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="loading">No failed runs found</td></tr>';
    return;
  }
  tbody.innerHTML = data.recent_errors.map(e => `
    <tr>
      <td><span class="cat-badge cat-${e.category_code}">${e.category_label}</span></td>
      <td><a class="run-link" href="/trace/${e.run_id}">${e.run_id.slice(0,12)}</a></td>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(e.task)}</td>
      <td class="err-text" title="${escapeHtml(e.error_message)}">${escapeHtml(e.error_message)}</td>
      <td>${e.latency_ms}ms</td>
      <td>${e.steps}</td>
    </tr>
  `).join('');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

loadData();
</script>
</body>
</html>
"""


@app.get("/errors", response_class=HTMLResponse)
def errors_page():
    return HTMLResponse(content=_ERRORS_HTML)
