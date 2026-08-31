"""Tests for token_efficiency analyzer."""

from __future__ import annotations

import pytest

from agent_eval.evaluation.token_efficiency import (
    BatchTokenAnalysis,
    RunTokenAnalysis,
    analyze_batch,
    analyze_run,
)
from agent_eval.trace.models import RunRecord, RunStatus, Span, SpanType, TokenUsage

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------


def _make_llm_span(span_id: str, trace_id: str, step: int, prompt: int, completion: int) -> Span:
    return Span(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=None,
        span_type=SpanType.LLM_CALL,
        step_index=step,
        name="gpt-4o-mini",
        input_data={"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}]},
        output_data={"content": "Hi there!"},
        tokens=TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
        cost=0.0,
        latency_ms=100,
        is_success=True,
    )


def _make_run(run_id: str, prompt: int, completion: int, final_output: str = "ok") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        status=RunStatus.SUCCESS,
        final_output=final_output,
        tokens=TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
    )


# ------------------------------------------------------------
# test_analyze_run_basic — 1 LLM span, check all fields populated
# ------------------------------------------------------------


def test_analyze_run_basic():
    run = _make_run("r_basic", prompt=200, completion=50, final_output="Hello world")
    spans = [_make_llm_span("s1", "r_basic", step=0, prompt=200, completion=50)]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert isinstance(analysis, RunTokenAnalysis)
    assert analysis.run_id == "r_basic"
    assert analysis.prompt_tokens == 200
    assert analysis.completion_tokens == 50
    assert analysis.total_tokens == 250
    assert analysis.final_answer_chars == 11  # len("Hello world")
    assert analysis.chars_per_completion_token == pytest.approx(11 / 50)
    # Only one LLM span → duplicates = 0
    assert analysis.duplicate_prompt_tokens == 0
    assert analysis.redundancy_ratio == 0.0
    assert analysis.context_window == 128_000
    assert analysis.context_used_pct == pytest.approx(200 / 128_000 * 100)
    assert isinstance(analysis.flags, list)


# ------------------------------------------------------------
# test_analyze_run_with_duplicates — 3 LLM spans simulating growing history
# ------------------------------------------------------------


def test_analyze_run_with_duplicates():
    run = _make_run("r_dup", prompt=500, completion=60, final_output="Final answer")
    # 3 LLM spans — each adds ~100 new tokens, final prompt grows to 500
    spans = [
        _make_llm_span("s1", "r_dup", step=0, prompt=100, completion=20),
        _make_llm_span("s2", "r_dup", step=1, prompt=300, completion=20),
        _make_llm_span("s3", "r_dup", step=2, prompt=500, completion=20),
    ]

    analysis = analyze_run(run, spans, context_window=128_000)

    # sum_llm_prompts = 100 + 300 + 500 = 900; run.tokens.prompt_tokens = 500
    # duplicates = 900 - 500 = 400
    assert analysis.duplicate_prompt_tokens == 400
    assert analysis.redundancy_ratio == pytest.approx(400 / 500)
    # high_redundancy flag should be present (0.8 > 0.25)
    assert "high_redundancy" in analysis.flags


# ------------------------------------------------------------
# test_context_near_limit_flag
# ------------------------------------------------------------


def test_context_near_limit_flag():
    # prompt=110000, window=128000 → context_used_pct = 85.9% → flag
    run = _make_run("r_ctx", prompt=110_000, completion=100, final_output="done")
    spans = [_make_llm_span("s1", "r_ctx", step=0, prompt=110_000, completion=100)]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.context_used_pct == pytest.approx(110_000 / 128_000 * 100)
    assert "context_near_limit" in analysis.flags


def test_context_not_near_limit():
    run = _make_run("r_ok_ctx", prompt=10_000, completion=100, final_output="done")
    spans = [_make_llm_span("s1", "r_ok_ctx", step=0, prompt=10_000, completion=100)]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert "context_near_limit" not in analysis.flags


# ------------------------------------------------------------
# test_high_redundancy_flag
# ------------------------------------------------------------


def test_high_redundancy_flag():
    # Build a run with exactly ~30% redundancy
    run = _make_run("r_hr", prompt=1_000, completion=50, final_output="ok")
    # sum prompts: 200 + 700 + 1000 = 1900 → duplicates = 900 → ratio 0.9
    spans = [
        _make_llm_span("s1", "r_hr", step=0, prompt=200, completion=20),
        _make_llm_span("s2", "r_hr", step=1, prompt=700, completion=20),
        _make_llm_span("s3", "r_hr", step=2, prompt=1000, completion=10),
    ]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.redundancy_ratio == pytest.approx(900 / 1000)
    assert "high_redundancy" in analysis.flags


def test_low_redundancy_no_flag():
    run = _make_run("r_lr", prompt=1_000, completion=50, final_output="ok")
    # Only 1 LLM call → 0 redundancy
    spans = [_make_llm_span("s1", "r_lr", step=0, prompt=1000, completion=50)]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.redundancy_ratio == 0.0
    assert "high_redundancy" not in analysis.flags


# ------------------------------------------------------------
# test_empty_completion_flag
# ------------------------------------------------------------


def test_empty_completion_flag():
    run = _make_run("r_ec", prompt=500, completion=0, final_output=None)
    spans = [_make_llm_span("s1", "r_ec", step=0, prompt=500, completion=0)]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.completion_tokens == 0
    assert "empty_completion" in analysis.flags


# ------------------------------------------------------------
# test_chars_per_token
# ------------------------------------------------------------


def test_chars_per_token():
    run = _make_run("r_cpt", prompt=100, completion=40, final_output="abcdefghij")  # 10 chars
    spans = [_make_llm_span("s1", "r_cpt", step=0, prompt=100, completion=40)]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.final_answer_chars == 10
    assert analysis.chars_per_completion_token == pytest.approx(10 / 40)


def test_chars_per_token_short_output_flags_low_efficiency():
    # completion=100 but only 50 chars → 0.5 < 1.0 → low_efficiency
    run = _make_run("r_le", prompt=200, completion=100, final_output="x" * 50)
    spans = [_make_llm_span("s1", "r_le", step=0, prompt=200, completion=100)]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.chars_per_completion_token == pytest.approx(0.5)
    assert "low_efficiency" in analysis.flags


# ------------------------------------------------------------
# test_analyze_batch — via conftest storage
# ------------------------------------------------------------


def test_analyze_batch(storage):
    # Run 1 — healthy, low redundancy
    run1 = _make_run("rb_1", prompt=800, completion=100, final_output="Hello world!")
    spans1 = [_make_llm_span("s1", "rb_1", step=0, prompt=800, completion=100)]
    storage.save_run(run1)
    storage.append_spans(spans1)

    # Run 2 — high redundancy
    run2 = _make_run("rb_2", prompt=600, completion=60, final_output="Another answer.")
    spans2 = [
        _make_llm_span("s1", "rb_2", step=0, prompt=200, completion=20),
        _make_llm_span("s2", "rb_2", step=1, prompt=400, completion=20),
        _make_llm_span("s3", "rb_2", step=2, prompt=600, completion=20),
    ]
    storage.save_run(run2)
    storage.append_spans(spans2)

    batch = analyze_batch(storage, context_window=128_000)

    assert isinstance(batch, BatchTokenAnalysis)
    assert len(batch.runs) == 2
    # Run 1: sum=800, final=800 → dup=0 → ratio 0
    # Run 2: sum=200+400+600=1200, final=600 → dup=600 → ratio 1.0
    assert batch.total_duplicate_tokens == 600
    assert batch.avg_redundancy_ratio == pytest.approx((0 + 1.0) / 2)
    # avg context used
    assert batch.avg_context_used_pct == pytest.approx(
        ((800 / 128_000 * 100) + (600 / 128_000 * 100)) / 2
    )
    # Top redundant runs sorted desc
    assert batch.top_redundant_runs[0][0] == "rb_2"
    assert batch.top_redundant_runs[0][1] == pytest.approx(1.0)
    assert batch.top_redundant_runs[1][0] == "rb_1"
    assert batch.top_redundant_runs[1][1] == pytest.approx(0.0)
    # Recommendations — run2 has high_redundancy, so should have recs
    assert isinstance(batch.recommendations, list)
    assert len(batch.recommendations) >= 1


def test_analyze_batch_empty(storage):
    batch = analyze_batch(storage, context_window=128_000)

    assert isinstance(batch, BatchTokenAnalysis)
    assert len(batch.runs) == 0
    assert batch.total_duplicate_tokens == 0
    assert batch.avg_redundancy_ratio == 0.0
    assert batch.recommendations == ["No significant token-efficiency issues detected. 🎉"]


def test_analyze_batch_with_run_ids(storage):
    run1 = _make_run("r1", prompt=500, completion=50, final_output="a")
    run2 = _make_run("r2", prompt=600, completion=60, final_output="b")
    s1 = [_make_llm_span("s1", "r1", step=0, prompt=500, completion=50)]
    s2 = [_make_llm_span("s1", "r2", step=0, prompt=600, completion=60)]
    storage.save_run(run1)
    storage.save_run(run2)
    storage.append_spans(s1)
    storage.append_spans(s2)

    # Only request r1
    batch = analyze_batch(storage, run_ids=["r1"], context_window=128_000)
    assert len(batch.runs) == 1
    assert batch.runs[0].run_id == "r1"


# ------------------------------------------------------------
# edge cases
# ------------------------------------------------------------


def test_analyze_run_no_llm_spans():
    """Run with zero LLM_CALL spans should produce zero-dup metrics safely."""
    run = _make_run("r_nollm", prompt=100, completion=20, final_output="empty")
    spans = [
        Span(
            span_id="s1",
            trace_id="r_nollm",
            span_type=SpanType.AGENT_STEP,
            step_index=0,
            tokens=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )
    ]

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.duplicate_prompt_tokens == 0
    assert analysis.redundancy_ratio == 0.0


def test_analyze_run_zero_prompt_tokens():
    """A run with prompt_tokens=0 must not divide by zero."""
    run = RunRecord(run_id="r_zero_prompt", tokens=TokenUsage(), final_output="")
    spans = []

    analysis = analyze_run(run, spans, context_window=128_000)

    assert analysis.prompt_tokens == 0
    assert analysis.redundancy_ratio == 0.0  # max(0, ...)/max(0,1) → 0.0
    assert analysis.chars_per_completion_token == 0.0
