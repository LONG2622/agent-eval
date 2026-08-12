"""Base evaluator interface and result models."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agent_eval.trace import RunRecord, Span


class EvalDimension(str, enum.Enum):
    SUCCESS_RATE = "success_rate"
    TOOL_USAGE = "tool_usage"
    ANSWER_QUALITY = "answer_quality"
    LATENCY = "latency"
    TOKEN_COST = "token_cost"


class SubMetric(str, enum.Enum):
    # Success
    PASS = "pass"
    # Tool usage
    TOOL_CALL_COUNT = "tool_call_count"
    TOOL_SUCCESS_RATE = "tool_success_rate"
    REDUNDANT_CALLS = "redundant_calls"
    # Quality
    KEYWORD_MATCH = "keyword_match"
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    # Latency
    TOTAL_LATENCY_MS = "total_latency_ms"
    AVG_STEP_LATENCY_MS = "avg_step_latency_ms"
    # Token / Cost
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    TOTAL_TOKENS = "total_tokens"
    TOTAL_COST = "total_cost"


class EvaluationResult(BaseModel):
    """Result from one evaluator for one run."""

    run_id: str
    evaluator: str
    dimension: EvalDimension
    sub_metric: SubMetric | None = None
    score: float = 0.0  # Normalized 0..1 or raw number, see details
    passed: bool | None = None
    details: str | list[str] | dict[str, Any] | None = None
    is_human: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_storage_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class BaseEvaluator(ABC):
    """Abstract evaluator - produces EvaluationResult(s) for a single run."""

    name: str
    dimension: EvalDimension

    @abstractmethod
    def evaluate(
        self,
        run: RunRecord,
        spans: list[Span],
    ) -> list[EvaluationResult]:
        """Evaluate one run. Return one or more result objects."""
        raise NotImplementedError
