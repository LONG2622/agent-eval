# 8.13 Phase 3 - FastAPI Server
"""Shared state for the agent-eval server (storage, engine, helpers)."""

from __future__ import annotations

from typing import Any

from agent_eval.config import load_config, reset_config
from agent_eval.evaluation import EvaluationEngine
from agent_eval.trace import JSONLStorage

# Force-reload config from .env to avoid stale cached values before storage init
reset_config()
_cfg = load_config(force_reload=True)

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
