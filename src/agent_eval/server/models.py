# 8.13 Phase 3 - FastAPI Server
"""Pydantic request models for the agent-eval server REST API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


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
