"""Token efficiency analysis for agent runs.

Detects prompt redundancy, context window utilization, conversation bloat,
and per-token output quality from trace spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_eval.config import load_config
from agent_eval.trace.models import RunRecord, Span, SpanType, TokenUsage
from agent_eval.trace.storage import JSONLStorage

DEFAULT_CONTEXT_WINDOW = 128_000


# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------


@dataclass
class RunTokenAnalysis:
    """Per-run token efficiency metrics."""

    run_id: str
    # Raw counts
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Efficiency
    final_answer_chars: int
    chars_per_completion_token: float
    # Redundancy
    duplicate_prompt_tokens: int
    redundancy_ratio: float
    # Context window utilization
    context_window: int
    context_used_pct: float
    # Prompt bloat
    system_prompt_tokens: int
    conversation_bloat_ratio: float
    # Verdict
    flags: list[str] = field(default_factory=list)


@dataclass
class BatchTokenAnalysis:
    """Aggregate token efficiency metrics across a batch of runs."""

    runs: list[RunTokenAnalysis]
    avg_redundancy_ratio: float
    avg_context_used_pct: float
    avg_chars_per_token: float
    total_duplicate_tokens: int
    top_redundant_runs: list[tuple[str, float]]  # (run_id, ratio) sorted desc
    recommendations: list[str] = field(default_factory=list)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _estimate_context_window(model: str | None) -> int:
    """Try to obtain context_window from config, fall back to default."""
    try:
        cfg = load_config()
        # TokenCostConfig has max_total_tokens which mirrors context window
        ctx = cfg.evaluation.token_cost.max_total_tokens
        if ctx and ctx > 0:
            return ctx
    except Exception:
        pass
    return DEFAULT_CONTEXT_WINDOW


def _collect_llm_spans(spans: list[Span]) -> list[Span]:
    """Filter and sort LLM_CALL spans by step_index."""
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM_CALL]
    llm_spans.sort(key=lambda s: s.step_index)
    return llm_spans


def _count_system_prompt_tokens(llm_spans: list[Span]) -> int:
    """Rough estimate: count tokens belonging to system messages.

    Since spans usually record aggregated TokenUsage per call, we use a
    heuristic: the first LLM call carries the full system prompt; subsequent
    calls re-send it verbatim. We count the system tokens only ONCE (from
    the first call) to get the unique system-prompt footprint.

    As a proxy we take the first LLM span's prompt tokens and subtract a
    minimal per-turn estimate (user + assistant tool result ≈ 50 tokens).
    """
    if not llm_spans:
        return 0
    first = llm_spans[0]
    # The first LLM call's prompt is roughly system + initial user message.
    # We assume system prompt is ~200 tokens minimum; if the first prompt
    # is small we return the whole thing as "system" (conservative).
    prompt = first.tokens.prompt_tokens
    if prompt <= 200:
        return prompt
    return prompt - 50  # 50 tokens reserved for first user turn


# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------


def analyze_run(
    run: RunRecord,
    spans: list[Span],
    context_window: int | None = None,
) -> RunTokenAnalysis:
    """Produce a RunTokenAnalysis for a single run.

    Args:
        run: The RunRecord to analyze.
        spans: All Span objects belonging to that run.
        context_window: Override the context window size; if None we try
            to read it from config, otherwise fall back to 128_000.
    """
    if context_window is None:
        context_window = _estimate_context_window(run.agent_config.get("model") if run.agent_config else None)

    prompt_tokens = run.tokens.prompt_tokens
    completion_tokens = run.tokens.completion_tokens
    total_tokens = run.tokens.total_tokens

    # 1. Efficiency
    final_answer_chars = len(run.final_output or "")
    chars_per_completion_token = final_answer_chars / max(completion_tokens, 1)

    # 2. Redundancy: sum of all LLM-call prompt tokens minus the run's
    #    final total prompt tokens (last LLM call re-sends full history).
    llm_spans = _collect_llm_spans(spans)
    sum_llm_prompts = sum(s.tokens.prompt_tokens for s in llm_spans)
    duplicate_prompt_tokens = max(0, sum_llm_prompts - prompt_tokens)
    redundancy_ratio = duplicate_prompt_tokens / max(prompt_tokens, 1)

    # 3. Context window utilization
    context_used_pct = (prompt_tokens / max(context_window, 1)) * 100

    # 4. Conversation bloat
    system_prompt_tokens = _count_system_prompt_tokens(llm_spans)
    conversation_bloat_ratio = system_prompt_tokens / max(prompt_tokens, 1)

    # 5. Flags
    flags: list[str] = []
    if redundancy_ratio > 0.25:
        flags.append("high_redundancy")
    if context_used_pct > 80.0:
        flags.append("context_near_limit")
    if completion_tokens > 0 and chars_per_completion_token < 1.0:
        flags.append("low_efficiency")
    if completion_tokens == 0:
        flags.append("empty_completion")
    if conversation_bloat_ratio > 0.10:
        flags.append("high_system_bloat")

    return RunTokenAnalysis(
        run_id=run.run_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        final_answer_chars=final_answer_chars,
        chars_per_completion_token=chars_per_completion_token,
        duplicate_prompt_tokens=duplicate_prompt_tokens,
        redundancy_ratio=redundancy_ratio,
        context_window=context_window,
        context_used_pct=context_used_pct,
        system_prompt_tokens=system_prompt_tokens,
        conversation_bloat_ratio=conversation_bloat_ratio,
        flags=flags,
    )


def analyze_batch(
    storage: JSONLStorage,
    run_ids: list[str] | None = None,
    context_window: int | None = None,
) -> BatchTokenAnalysis:
    """Analyze one or more runs from storage and produce aggregate metrics.

    Args:
        storage: A JSONLStorage instance used to discover runs + spans.
        run_ids: Specific run IDs to include. If None, all runs are analysed.
        context_window: Override the context window for every run.
    """
    if run_ids is None:
        runs = storage.list_runs()
    else:
        runs = []
        for rid in run_ids:
            r = storage.load_run(rid)
            if r is not None:
                runs.append(r)

    if not runs:
        return BatchTokenAnalysis(
            runs=[],
            avg_redundancy_ratio=0.0,
            avg_context_used_pct=0.0,
            avg_chars_per_token=0.0,
            total_duplicate_tokens=0,
            top_redundant_runs=[],
            recommendations=["No significant token-efficiency issues detected. 🎉"],
        )

    run_analyses: list[RunTokenAnalysis] = []
    for run in runs:
        spans = storage.load_spans(run.run_id)
        run_analyses.append(analyze_run(run, spans, context_window=context_window))

    n = len(run_analyses)
    avg_redundancy_ratio = sum(r.redundancy_ratio for r in run_analyses) / n
    avg_context_used_pct = sum(r.context_used_pct for r in run_analyses) / n
    avg_chars_per_token = sum(r.chars_per_completion_token for r in run_analyses) / n
    total_duplicate_tokens = sum(r.duplicate_prompt_tokens for r in run_analyses)

    top_redundant = sorted(run_analyses, key=lambda r: r.redundancy_ratio, reverse=True)
    top_redundant_runs = [(r.run_id, r.redundancy_ratio) for r in top_redundant[:10]]

    recommendations = _build_recommendations(run_analyses)

    return BatchTokenAnalysis(
        runs=run_analyses,
        avg_redundancy_ratio=avg_redundancy_ratio,
        avg_context_used_pct=avg_context_used_pct,
        avg_chars_per_token=avg_chars_per_token,
        total_duplicate_tokens=total_duplicate_tokens,
        top_redundant_runs=top_redundant_runs,
        recommendations=recommendations,
    )


# ------------------------------------------------------------
# Recommendation engine
# ------------------------------------------------------------


def _build_recommendations(run_analyses: list[RunTokenAnalysis]) -> list[str]:
    """Convert observed flags into human-friendly recommendations."""
    recs: list[str] = []
    n = len(run_analyses)
    if n == 0:
        return recs

    high_red = [r for r in run_analyses if "high_redundancy" in r.flags]
    near_limit = [r for r in run_analyses if "context_near_limit" in r.flags]
    low_eff = [r for r in run_analyses if "low_efficiency" in r.flags]
    sys_bloat = [r for r in run_analyses if "high_system_bloat" in r.flags]

    if len(high_red) / n > 0.3:
        recs.append(
            "Consider adding conversation summarization: "
            f"{len(high_red)}/{n} runs have >25% prompt redundancy."
        )
    if near_limit:
        recs.append(
            f"{len(near_limit)} run(s) approached the context window limit "
            f"(>80%). Consider switching to a larger-context model or adding "
            "proactive truncation."
        )
    if sys_bloat:
        recs.append(
            f"{len(sys_bloat)} run(s) show high system-prompt bloat. "
            "Review the system prompt and trim rarely-used tool descriptions."
        )
    if low_eff:
        recs.append(
            f"{len(low_eff)} run(s) have low chars-per-completion-token ratio, "
            "suggesting fluff or tool-loop behaviour. Inspect those traces."
        )
    if not recs:
        recs.append("No significant token-efficiency issues detected. 🎉")

    return recs


# Re-export TokenUsage for convenience (consumers may import from here)
__all__ = [
    "RunTokenAnalysis",
    "BatchTokenAnalysis",
    "analyze_run",
    "analyze_batch",
    "DEFAULT_CONTEXT_WINDOW",
    "TokenUsage",
]
