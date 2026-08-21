"""Trace data models - RunRecord and Span schemas."""

from __future__ import annotations

import enum
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class SpanType(str, enum.Enum):
    """Category of a trace span."""

    AGENT_STEP = "agent_step"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    THOUGHT = "thought"


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_pair(cls, prompt: int, completion: int) -> "TokenUsage":
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class Span(BaseModel):
    """A single trace event - one step in the execution graph."""

    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str  # same as run_id
    parent_span_id: str | None = None
    span_type: SpanType
    step_index: int = 0
    name: str = ""  # model name, tool name, etc.
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost: float = 0.0
    latency_ms: int = 0
    is_success: bool = True
    exception: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_timestamp)

    def to_storage_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RunRecord(BaseModel):
    """High-level record for one agent execution (one trace)."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    task_id: str = ""  # user-provided task reference
    agent_name: str = ""
    agent_config: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    input_text: str = ""
    final_output: str | None = None
    total_steps: int = 0
    total_latency_ms: int = 0
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    total_cost: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None  # for evaluation
    ground_truth: dict[str, Any] | None = None
    started_at: str = Field(default_factory=_utc_timestamp)
    finished_at: str | None = None

    @property
    def trace_id(self) -> str:
        return self.run_id

    def mark_finished(self, status: RunStatus, *, error: str | None = None) -> None:
        self.status = status
        self.error_message = error
        self.finished_at = _utc_timestamp()

    def to_storage_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ============================================================
# Annotation Record (for human annotation)
# ============================================================


class AnnotationRecord(BaseModel):
    """Human annotation for a run."""

    annotation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    run_id: str
    annotator: str = "anonymous"
    score: int = Field(..., ge=1, le=5, description="Human score 1-5")
    labels: list[str] = Field(default_factory=list)
    comment: str = ""
    created_at: str = Field(default_factory=_utc_timestamp)

    def to_storage_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
