# 8.13 Phase 3 - FastAPI Server
"""Pydantic request models for the agent-eval server REST API."""

from __future__ import annotations

from pydantic import BaseModel


class RunTaskRequest(BaseModel):
    task: str
    agent_type: str = "react"
    model: str | None = None
    temperature: float | None = None
    max_steps: int = 10
    expected_output: str | None = None
    task_id: str | None = None


class EvalBatchRequest(BaseModel):
    dataset_path: str
    agent_type: str = "react"
    model: str | None = None
    temperature: float | None = None
    max_steps: int = 10
    sample: int = 0
    workers: int = 1
    retries: int = 2


class AnnotationRequest(BaseModel):
    score: int = 5
    labels: list[str] = []
    comment: str = ""
    annotator: str = "anonymous"
