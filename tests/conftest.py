"""Test configuration and shared fixtures for Agent Eval test suite."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the src directory is in Python path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@pytest.fixture
def tmp_output_dir() -> Path:
    """Create a temporary output directory for test data."""
    with tempfile.TemporaryDirectory(prefix="agent_test_") as tmp:
        output_dir = Path(tmp) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "traces").mkdir(exist_ok=True)
        (output_dir / "runs").mkdir(exist_ok=True)
        (output_dir / "evaluations").mkdir(exist_ok=True)
        (output_dir / "annotations").mkdir(exist_ok=True)
        yield output_dir


@pytest.fixture
def patch_config(monkeypatch, tmp_output_dir):
    """Patch the config to use temporary directories."""
    from agent_eval import config
    config.reset_config()

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_output_dir))
    monkeypatch.setenv("TRACE_DIR", str(tmp_output_dir / "traces"))
    monkeypatch.setenv("RUN_DIR", str(tmp_output_dir / "runs"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_output_dir / "evaluations"))

    # Force reload config with fresh env
    cfg = config.load_config(force_reload=True)
    yield cfg
    config.reset_config()


@pytest.fixture
def sample_run():
    """Create a sample RunRecord for testing."""
    from agent_eval.trace.models import RunRecord, RunStatus, TokenUsage

    run = RunRecord(
        run_id="test_run_001",
        task_id="task_001",
        agent_name="react_agent",
        input_text="Calculate sqrt(144) + 5^2",
        status=RunStatus.SUCCESS,
        final_output="37",
        total_steps=3,
        total_latency_ms=5000,
        tokens=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        total_cost=0.0003,
        expected_output="37",
    )
    return run


@pytest.fixture
def sample_spans():
    """Create sample spans for testing."""
    from agent_eval.trace.models import Span, SpanType, TokenUsage

    spans = [
        Span(
            span_id="span_001",
            trace_id="test_run_001",
            span_type=SpanType.AGENT_STEP,
            step_index=0,
            name="agent_step",
            input_data={"task": "Calculate sqrt(144) + 5^2"},
            output_data={"thought": "I need to calculate this expression"},
            tokens=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            cost=0.0001,
            latency_ms=1000,
            is_success=True,
        ),
        Span(
            span_id="span_002",
            trace_id="test_run_001",
            span_type=SpanType.LLM_CALL,
            step_index=1,
            name="gpt-4o-mini",
            input_data={"messages": [{"role": "user", "content": "Calculate sqrt(144) + 5^2"}]},
            output_data={"content": "I'll use the calculator tool."},
            tokens=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
            cost=0.00005,
            latency_ms=2000,
            is_success=True,
        ),
        Span(
            span_id="span_003",
            trace_id="test_run_001",
            span_type=SpanType.TOOL_CALL,
            step_index=2,
            name="calculator",
            input_data={"arguments": {"expression": "sqrt(144) + 5^2"}},
            output_data={"result": "37.0"},
            tokens=TokenUsage(),
            cost=0.0,
            latency_ms=500,
            is_success=True,
        ),
        Span(
            span_id="span_004",
            trace_id="test_run_001",
            span_type=SpanType.AGENT_STEP,
            step_index=3,
            name="agent_step",
            input_data={"observation": "37.0"},
            output_data={"final_answer": "37"},
            tokens=TokenUsage(prompt_tokens=20, completion_tokens=20, total_tokens=40),
            cost=0.00015,
            latency_ms=1500,
            is_success=True,
        ),
    ]
    return spans


@pytest.fixture
def storage(tmp_output_dir):
    """Create a JSONLStorage with temporary directories."""
    from agent_eval.trace.storage import JSONLStorage

    return JSONLStorage(
        trace_dir=tmp_output_dir / "traces",
        run_dir=tmp_output_dir / "runs",
        annotation_dir=tmp_output_dir / "annotations",
    )


@pytest.fixture
def annotated_run_data(storage, sample_run, sample_spans):
    """Save a run with spans and an annotation, and return them."""
    from agent_eval.trace.models import AnnotationRecord

    # Save the run
    storage.save_run(sample_run)

    # Save spans
    storage.append_spans(sample_spans)

    # Create annotation
    ann = AnnotationRecord(
        run_id="test_run_001",
        annotator="test_annotator",
        score=4,
        labels=["correct", "complete"],
        comment="Good answer, correct result",
    )
    storage.save_annotation(ann)

    return sample_run, sample_spans, ann
