# 8.13 Phase 3 - FastAPI Server
"""HTML page endpoints for the agent-eval server dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

# Templates are read once at module load time from ../templates
_template_dir = Path(__file__).parent.parent / "templates"

_DASHBOARD_HTML = (_template_dir / "dashboard.html").read_text(encoding="utf-8")
_TRACE_HTML = (_template_dir / "trace.html").read_text(encoding="utf-8")
_ANNOTATE_HTML = (_template_dir / "annotate.html").read_text(encoding="utf-8")
_CHAT_HTML = (_template_dir / "chat.html").read_text(encoding="utf-8")
_ERRORS_HTML = (_template_dir / "errors.html").read_text(encoding="utf-8")
_COMPARE_HTML = (_template_dir / "compare.html").read_text(encoding="utf-8")

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
@router.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


@router.get("/trace/{run_id}", response_class=HTMLResponse)
def trace_page(run_id: str):
    return HTMLResponse(content=_TRACE_HTML)


@router.get("/annotate/{run_id}", response_class=HTMLResponse)
def annotate_page(run_id: str):
    return HTMLResponse(content=_ANNOTATE_HTML)


@router.get("/chat", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(content=_CHAT_HTML)


@router.get("/errors", response_class=HTMLResponse)
def errors_page():
    return HTMLResponse(content=_ERRORS_HTML)


@router.get("/compare", response_class=HTMLResponse)
def compare_page():
    return HTMLResponse(content=_COMPARE_HTML)
