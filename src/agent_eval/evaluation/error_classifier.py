"""Error classifier - categorizes FAILED runs by error type."""

# 8.13
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agent_eval.trace.storage import JSONLStorage


@dataclass
class ErrorCategory:
    """One category of errors."""
    code: str
    label: str
    description: str
    patterns: list[str] = field(default_factory=list)

    def matches(self, error_text: str) -> bool:
        text = error_text.lower()
        return any(p in text for p in self.patterns)


# ── Error categories (checked in order, first match wins) ──────────────

CATEGORIES: list[ErrorCategory] = [
    ErrorCategory(
        code="llm_timeout",
        label="LLM Timeout",
        description="LLM API request timed out",
        patterns=[
            "apitimeouterror",
            "request timed out",
            "timeout",
            "timed out",
        ],
    ),
    ErrorCategory(
        code="llm_rate_limit",
        label="LLM Rate Limit",
        description="LLM API rate limit or quota exceeded",
        patterns=[
            "rate limit",
            "rate_limit",
            "quota",
            "429",
            "too many requests",
        ],
    ),
    ErrorCategory(
        code="file_system_error",
        label="File System Error",
        description="File I/O or permission error (Windows file lock, disk write)",
        patterns=[
            "permissionerror",
            "winerror",
            "errno 13",
            "file not found",
            "no such file",
            "being used by another process",
            "runs.tmp",
            "traces.tmp",
            "evaluations.tmp",
        ],
    ),
    ErrorCategory(
        code="llm_auth_error",
        label="LLM Auth Error",
        description="API key invalid or unauthorized",
        patterns=[
            "authenticationerror",
            "unauthorized",
            "invalid api key",
            "401",
            "403",
        ],
    ),
    ErrorCategory(
        code="llm_bad_request",
        label="LLM Bad Request",
        description="LLM rejected the request (format, params, model name)",
        patterns=[
            "badrequesterror",
            "400",
            "validation error",
            "invalid model",
            "model not found",
            "invalid request",
        ],
    ),
    ErrorCategory(
        code="llm_not_found",
        label="LLM Endpoint Not Found",
        description="API endpoint returned 404",
        patterns=[
            "404",
            "not found",
            "page not found",
        ],
    ),
    ErrorCategory(
        code="tool_execution_error",
        label="Tool Execution Error",
        description="A tool call raised an exception",
        patterns=[
            "tool",
            "calculator",
            "web_search",
            "get_time",
            "read_file",
        ],
    ),
    ErrorCategory(
        code="llm_format_error",
        label="LLM Response Format Error",
        description="LLM response could not be parsed (JSON, function call)",
        patterns=[
            "json",
            "parse",
            "decode",
            "unexpected token",
            "invalid function",
            "tool_calls",
        ],
    ),
    ErrorCategory(
        code="max_steps_exceeded",
        label="Max Steps Exceeded",
        description="Agent exceeded the maximum reasoning steps",
        patterns=[
            "max_steps",
            "maximum steps",
            "step limit",
            "too many steps",
        ],
    ),
    ErrorCategory(
        code="network_error",
        label="Network Error",
        description="Network connectivity issue",
        patterns=[
            "connectionerror",
            "connection error",
            "connection refused",
            "connection reset",
            "connection aborted",
            "network",
            "unreachable",
            "ssl",
            "certificate",
        ],
    ),
    ErrorCategory(
        code="internal_error",
        label="Internal Code Error",
        description="Bug in the evaluation framework itself",
        patterns=[
            "typeerror",
            "attributeerror",
            "keyerror",
            "valueerror",
            "indexerror",
            "runtimeerror",
            "does not support the context manager",
            "object has no attribute",
        ],
    ),
]

UNKNOWN = ErrorCategory(
    code="unknown",
    label="Unknown Error",
    description="Unclassified error - check raw error message",
)


def classify_error(error_message: str, span_data: list[dict] | None = None) -> ErrorCategory:
    """Classify a single error message.

    Args:
        error_message: The error_message field from a RunRecord.
        span_data: Optional list of span dicts to check for tool errors.

    Returns:
        The matching ErrorCategory, or UNKNOWN.
    """
    if not error_message:
        # If no error message but spans show failure, check span errors
        if span_data:
            for span in span_data:
                if not span.get("is_success", True):
                    err = span.get("error_data") or span.get("output_data", {})
                    if isinstance(err, dict):
                        err = str(err)
                    cat = _match_first(str(err))
                    if cat.code != "unknown":
                        return cat
        return UNKNOWN
    return _match_first(error_message)


def _match_first(text: str) -> ErrorCategory:
    """Find the first matching category."""
    for cat in CATEGORIES:
        if cat.matches(text):
            return cat
    return UNKNOWN


@dataclass
class ClassifiedError:
    """A single classified error instance."""
    run_id: str
    task: str
    agent_name: str
    category: ErrorCategory
    error_message: str
    latency_ms: int
    steps: int


@dataclass
class ErrorSummary:
    """Aggregated error summary."""
    total_runs: int = 0
    total_failed: int = 0
    total_success: int = 0
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_errors: list[ClassifiedError] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.total_failed / self.total_runs if self.total_runs > 0 else 0.0


def classify_all_runs(storage: JSONLStorage, limit: int = 0) -> ErrorSummary:
    """Classify all failed runs in storage.

    Args:
        storage: JSONLStorage instance.
        limit: Max number of recent errors to include (0 = all).

    Returns:
        ErrorSummary with aggregated stats.
    """
    all_runs = storage.list_runs()
    summary = ErrorSummary(
        total_runs=len(all_runs),
        total_failed=sum(1 for r in all_runs if r.status.value == "failed"),
        total_success=sum(1 for r in all_runs if r.status.value == "success"),
    )

    cat_counter: Counter[str] = Counter()

    for run in all_runs:
        if run.status.value != "failed":
            continue

        # Load spans for richer error context
        spans = storage.load_spans(run.run_id)
        span_dicts = [s.to_storage_dict() for s in spans] if spans else None

        category = classify_error(run.error_message or "", span_dicts)
        cat_counter[category.code] += 1

        classified = ClassifiedError(
            run_id=run.run_id,
            task=run.input_text or run.task_id,
            agent_name=run.agent_name,
            category=category,
            error_message=run.error_message or "(no error message)",
            latency_ms=run.total_latency_ms,
            steps=run.total_steps,
        )
        summary.recent_errors.append(classified)

    # Build category stats
    for cat in CATEGORIES + [UNKNOWN]:
        count = cat_counter.get(cat.code, 0)
        if count > 0:
            summary.by_category[cat.code] = {
                "code": cat.code,
                "label": cat.label,
                "description": cat.description,
                "count": count,
                "percentage": round(count / summary.total_failed * 100, 1) if summary.total_failed > 0 else 0,
            }

    # Sort recent errors by most recent first
    summary.recent_errors.reverse()

    if limit > 0:
        summary.recent_errors = summary.recent_errors[:limit]

    return summary


def format_summary_text(summary: ErrorSummary) -> str:
    """Format error summary as readable text for CLI output."""
    lines = []
    lines.append(f"Total Runs: {summary.total_runs}")
    lines.append(f"  Success: {summary.total_success}")
    lines.append(f"  Failed:  {summary.total_failed}")
    lines.append(f"  Failure Rate: {summary.failure_rate:.1%}")
    lines.append("")
    lines.append("Error Distribution:")
    lines.append("-" * 60)

    if not summary.by_category:
        lines.append("  (no failed runs to classify)")
    else:
        # Sort by count descending
        cats = sorted(summary.by_category.values(), key=lambda x: x["count"], reverse=True)
        for cat in cats:
            lines.append(f"  {cat['label']:<25} {cat['count']:>4} ({cat['percentage']}%)")
            lines.append(f"    └─ {cat['description']}")

    lines.append("")
    lines.append(f"Recent Failed Runs ({len(summary.recent_errors)}):")
    lines.append("-" * 60)
    for err in summary.recent_errors[:15]:
        lines.append(f"  [{err.category.label}] {err.run_id}")
        lines.append(f"    Task: {err.task[:70]}")
        lines.append(f"    Error: {err.error_message[:100]}")
        lines.append(f"    Latency: {err.latency_ms}ms, Steps: {err.steps}")
        lines.append("")

    return "\n".join(lines)
