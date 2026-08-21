# 8.13 Phase 3 - FastAPI Server
"""FastAPI server exposing agent-eval capabilities via REST API + Web Dashboard.

This module wires up the FastAPI application, CORS middleware, and includes all
routers defined under ``server.routes``. Shared state lives in ``server.state``
and request models in ``server.models``; HTML templates are loaded from
``server/templates`` by ``routes/pages.py``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_eval import __version__
from agent_eval.server.routes import api, pages, websocket

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

# Include all route modules
app.include_router(api.router)
app.include_router(pages.router)
app.include_router(websocket.router)
