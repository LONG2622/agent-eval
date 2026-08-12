"""Offline smoke tests for Phase 1 MVP (no real LLM API needed)."""
import sys, tempfile, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
tmp = tempfile.mkdtemp()
os.environ["OUTPUT_DIR"] = tmp + "/outputs"
os.environ["TRACE_DIR"] = tmp + "/outputs/traces"
os.environ["RUN_DIR"] = tmp + "/outputs/runs"
os.environ["EVAL_DIR"] = tmp + "/outputs/evaluations"

def main():
    # -------- Test 1: Tool Registry + Builtins --------
    from agent_eval.tools import ToolRegistry, register_builtin_tools
    reg = ToolRegistry()
    register_builtin_tools(reg)
    tools = reg.list_tools()
    print("Test 1 - ToolRegistry: {} tools registered = {}".format(len(tools), [t.name for t in tools]))
    assert len(tools) == 4

    r = reg.invoke("calculator", {"expression": "sqrt(144) + 5^2"})
    print("  calculator sqrt(144)+5^2 -> output={} ok={} time={}ms".format(r.output, r.success, r.latency_ms))
    assert r.success and "37" in r.output

    r = reg.invoke("web_search", {"query": "What is PyTorch"})
    print("  web_search PyTorch -> success={} len={}".format(r.success, len(r.output)))
    assert r.success and "Meta" in r.output

    # -------- Test 2: TraceRecorder + JSONLStorage --------
    from agent_eval.config import load_config
    load_config()
    from agent_eval.trace import (
        TraceRecorder, JSONLStorage, RunRecord, Span,
        SpanType, TokenUsage, RunStatus,
    )
    storage = JSONLStorage()
    rec = TraceRecorder(storage)

    run = RunRecord(
        task_id="test_001",
        agent_name="react-test",
        input_text="Calculate sqrt(144) + 5^2",
        expected_output="sqrt(144)=12, 5^2=25, result=37",
    )
    rec.start_run(run)
    rec.on_step_start(run, 1, "I need to compute sqrt(144) + 5^2. Let me use the calculator.")
    sp = Span(
        trace_id=run.trace_id, span_type=SpanType.LLM_CALL, step_index=1, name="gpt-4o-mini",
        tokens=TokenUsage.from_pair(50, 30), latency_ms=850, cost=0.0001,
        output_data={"content": "Let me call calculator", "tool_calls": [{"name": "calculator"}]},
    )
    rec._add_span(sp)
    sp2 = Span(
        trace_id=run.trace_id, span_type=SpanType.TOOL_CALL, step_index=1, name="calculator",
        input_data={"arguments": {"expression": "sqrt(144)+5^2"}},
        output_data={"output": "37.0", "success": True}, latency_ms=5, is_success=True,
    )
    rec._add_span(sp2)
    rec.on_step_end(run, 1, "calculator(sqrt(144)+5^2)", "37.0")
    final_run = rec.end_run(run, status=RunStatus.SUCCESS, final_output="The result is 37")
    print(
        "Test 2 - TraceRecorder: run={} steps={} cost=${:.4f} tokens={}".format(
            final_run.run_id, final_run.total_steps, final_run.total_cost, final_run.tokens.total_tokens
        )
    )
    assert final_run.status == RunStatus.SUCCESS
    assert final_run.total_cost > 0

    # -------- Test 3: Storage round-trip --------
    loaded_run = storage.load_run(final_run.run_id)
    loaded_spans = storage.load_spans(final_run.trace_id)
    print("Test 3 - Storage: loaded_run ok={} spans_count={}".format(loaded_run is not None, len(loaded_spans)))
    assert loaded_run and len(loaded_spans) >= 2

    # -------- Test 4: Evaluation Engine --------
    from agent_eval.evaluation import EvaluationEngine
    engine = EvaluationEngine(storage=storage)
    results = engine.evaluate_run(final_run.run_id)
    dims = set(r.dimension.value for r in results)
    print("Test 4 - Evaluation: {} scores, dimensions={}".format(len(results), dims))
    assert {"success_rate", "tool_usage", "answer_quality", "latency", "token_cost"} <= dims

    # -------- Test 5: TaskDataset --------
    from agent_eval.task import TaskDataset
    sample_tasks = [
        {"task_id": "t1", "input": "What is 2+2", "expected_output": "4"},
        {"task_id": "t2", "input": "Capital of France", "expected_output": "Paris"},
    ]
    ds = TaskDataset.from_list(sample_tasks, name="tiny")
    print("Test 5 - TaskDataset: {} items, ids=[{}, {}]".format(len(ds), ds.items[0].task_id, ds.items[1].task_id))
    assert len(ds) == 2

    # -------- Test 6: Batch evaluation + aggregate --------
    # Simulate another run
    run2 = RunRecord(
        task_id="test_002", agent_name="react-test",
        input_text="Capital of France", expected_output="Paris",
    )
    rec2 = TraceRecorder(storage)
    rec2.start_run(run2)
    rec2.on_step_start(run2, 1, "I should search this.")
    rec2._add_span(Span(
        trace_id=run2.trace_id, span_type=SpanType.LLM_CALL, step_index=1, name="gpt-4o-mini",
        tokens=TokenUsage.from_pair(80, 15), latency_ms=500, cost=0.00005,
        output_data={"content": "Paris"},
    ))
    rec2.on_step_end(run2, 1, "direct_answer", "Paris")
    rec2.end_run(run2, status=RunStatus.SUCCESS, final_output="The capital is Paris")

    per_run, summary = engine.evaluate_runs([final_run.run_id, run2.run_id], save_summary=False)
    print(
        "Test 6 - Batch aggregation: runs={} success_rate={:.1%} quality={:.2f} avg_latency={:.0f}ms total_cost=${:.5f}".format(
            summary.evaluated_runs, summary.overall_success_rate,
            summary.overall_quality_score, summary.avg_latency_ms, summary.total_cost,
        )
    )
    assert summary.evaluated_runs == 2
    assert summary.total_cost > 0

    print()
    print("=" * 60)
    print("ALL 6 OFFLINE FUNCTIONAL TESTS PASSED (MVP Phase 1)")
    print("=" * 60)

if __name__ == "__main__":
    main()
