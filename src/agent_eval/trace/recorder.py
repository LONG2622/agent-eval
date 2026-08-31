"""Trace Recorder - central hub that collects and persists spans.

Also implements LLMCallback / ToolCallback / AgentCallback so it can be
plugged directly into LLMGateway, ToolRegistry, and AgentRuntime.
"""

from __future__ import annotations

import logging
import time

from agent_eval.llm import LLMCallback, LLMCallContext
from agent_eval.tools import ToolCallback, ToolCallContext
from agent_eval.trace.models import RunRecord, RunStatus, Span, SpanType, TokenUsage
from agent_eval.trace.storage import JSONLStorage

logger = logging.getLogger("agent_eval.trace.recorder")


class AgentCallback:
    """Hook for agent-level lifecycle events (step start/end, session start/end)."""

    def on_agent_start(self, run: RunRecord) -> None:
        pass

    def on_agent_end(self, run: RunRecord) -> None:
        pass

    def on_step_start(self, run: RunRecord, step_index: int, thought: str) -> None:
        pass

    def on_step_end(self, run: RunRecord, step_index: int, action: str, observation: str) -> None:
        pass


class TraceRecorder(LLMCallback, ToolCallback, AgentCallback):
    """Collects spans from all sources and writes them to storage.

    Threading note: MVP is single-threaded per run. We don't use a background
    queue yet - spans flush to storage immediately when a run ends.
    """

    def __init__(self, storage: JSONLStorage | None = None) -> None:
        self._storage = storage or JSONLStorage()
        self._active_run: RunRecord | None = None
        self._pending_spans: list[Span] = []
        self._current_agent_step_span_id: str | None = None

    @property
    def storage(self) -> JSONLStorage:
        return self._storage

    # ================ Agent Lifecycle ================

    def start_run(self, run: RunRecord) -> RunRecord:
        """Mark the start of a new agent run."""
        run.status = RunStatus.RUNNING
        self._active_run = run
        self._pending_spans.clear()
        self._storage.save_run(run)
        logger.info(f"Started run {run.run_id} (task={run.task_id}, agent={run.agent_name})")
        return run

    def end_run(
        self,
        run: RunRecord,
        *,
        status: RunStatus,
        final_output: str | None = None,
        error: str | None = None,
    ) -> RunRecord:
        """Finalize a run, persist all spans and run metadata."""
        run.final_output = final_output
        run.total_steps = max(run.total_steps, 1 if self._pending_spans else 0)
        run.mark_finished(status, error=error)

        # Aggregate totals from spans
        tokens = TokenUsage()
        total_cost = 0.0
        total_latency = 0
        steps = 0
        for span in self._pending_spans:
            tokens = tokens.add(span.tokens)
            total_cost += span.cost
            total_latency += span.latency_ms
            if span.span_type == SpanType.AGENT_STEP:
                steps += 1
        run.tokens = tokens
        run.total_cost = round(total_cost, 6)
        run.total_steps = steps or run.total_steps
        # If we have spans, total latency is from first to last
        if self._pending_spans:
            try:
                from datetime import datetime

                tss = [
                    datetime.fromisoformat(s.created_at.replace("Z", "+00:00"))
                    for s in self._pending_spans
                ]
                if tss:
                    run.total_latency_ms = int((max(tss) - min(tss)).total_seconds() * 1000)
            except (RuntimeError, ValueError, TypeError):
                run.total_latency_ms = total_latency

        # Persist
        self._storage.append_spans(self._pending_spans)
        self._storage.save_run(run)
        self._pending_spans.clear()
        self._active_run = None
        self._current_agent_step_span_id = None
        logger.info(
            f"Finished run {run.run_id}: {status.value} in "
            f"{run.total_latency_ms}ms, ${run.total_cost:.4f}"
        )
        return run

    # ================ Span Factory ================

    def _add_span(self, span: Span) -> None:
        if self._active_run is None:
            logger.debug(f"No active run; dropping span {span.span_type}/{span.name}")
            return
        span.trace_id = self._active_run.trace_id
        self._pending_spans.append(span)

    # ================ AgentCallback ================

    def on_agent_start(self, run: RunRecord) -> None:
        self.start_run(run)

    def on_agent_end(self, run: RunRecord) -> None:
        # end_run is explicitly called; noop here
        pass

    def on_step_start(self, run: RunRecord, step_index: int, thought: str) -> None:
        self._active_run = run
        span = Span(
            trace_id=run.trace_id,
            parent_span_id=None,
            span_type=SpanType.AGENT_STEP,
            step_index=step_index,
            name=f"step_{step_index}",
            input_data={"thought": thought},
            metadata={"phase": "start"},
        )
        self._current_agent_step_span_id = span.span_id
        self._add_span(span)

    def on_step_end(self, run: RunRecord, step_index: int, action: str, observation: str) -> None:
        self._active_run = run
        span = Span(
            trace_id=run.trace_id,
            parent_span_id=self._current_agent_step_span_id,
            span_type=SpanType.AGENT_STEP,
            step_index=step_index,
            name=f"step_{step_index}_end",
            output_data={"action": action, "observation": observation},
            metadata={"phase": "end"},
        )
        self._add_span(span)

    # ================ LLMCallback ================

    def on_call_start(self, ctx: LLMCallContext) -> None:
        pass  # record on end only (no latency / tokens yet)

    def on_call_end(self, ctx: LLMCallContext) -> None:
        if self._active_run is None or ctx.response is None:
            return
        resp = ctx.response
        cost = getattr(resp, "_cost", 0.0) or 0.0
        run = self._active_run
        span = Span(
            trace_id=run.trace_id,
            parent_span_id=self._current_agent_step_span_id,
            span_type=SpanType.LLM_CALL,
            step_index=run.total_steps,
            name=resp.model or ctx.model,
            input_data={
                "messages": [m.model_dump(mode="json") for m in ctx.messages],
                "temperature": ctx.options.temperature,
                "max_tokens": ctx.options.max_tokens,
            },
            output_data={
                "content": resp.content,
                "tool_calls": resp.tool_calls,
            },
            tokens=TokenUsage.from_pair(resp.prompt_tokens, resp.completion_tokens),
            cost=cost,
            latency_ms=resp.latency_ms,
            is_success=True,
        )
        self._add_span(span)

    def on_call_error(self, ctx: LLMCallContext) -> None:
        if self._active_run is None:
            return
        run = self._active_run
        err = ctx.error
        span = Span(
            trace_id=run.trace_id,
            parent_span_id=self._current_agent_step_span_id,
            span_type=SpanType.LLM_CALL,
            step_index=run.total_steps,
            name=ctx.model,
            input_data={
                "messages": [m.model_dump(mode="json") for m in ctx.messages],
            },
            latency_ms=int((time.time() - ctx.started_at) * 1000),
            is_success=False,
            exception=f"{type(err).__name__}: {err}" if err else "Unknown error",
        )
        self._add_span(span)

    # ================ ToolCallback ================

    def on_tool_start(self, ctx: ToolCallContext) -> None:
        pass

    def on_tool_end(self, ctx: ToolCallContext) -> None:
        if self._active_run is None or ctx.result is None:
            return
        run = self._active_run
        result = ctx.result
        span = Span(
            trace_id=run.trace_id,
            parent_span_id=self._current_agent_step_span_id,
            span_type=SpanType.TOOL_CALL,
            step_index=run.total_steps,
            name=ctx.tool_name,
            input_data={"arguments": ctx.arguments},
            output_data={
                "output": result.output,
                "success": result.success,
                "error": result.error,
            },
            latency_ms=result.latency_ms,
            is_success=result.success,
            exception=None if result.success else (result.error or "Tool failed"),
        )
        self._add_span(span)

    def on_tool_error(self, ctx: ToolCallContext) -> None:
        if self._active_run is None:
            return
        run = self._active_run
        err = ctx.error
        span = Span(
            trace_id=run.trace_id,
            parent_span_id=self._current_agent_step_span_id,
            span_type=SpanType.TOOL_CALL,
            step_index=run.total_steps,
            name=ctx.tool_name,
            input_data={"arguments": ctx.arguments},
            latency_ms=int((time.time() - ctx.started_at) * 1000),
            is_success=False,
            exception=f"{type(err).__name__}: {err}" if err else "Tool error",
        )
        self._add_span(span)
