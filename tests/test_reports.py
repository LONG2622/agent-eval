"""Tests for report rendering: rich terminal output and HTML report generation.

All tests are offline - only the 5 built-in heuristic evaluators are used, no LLM.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from agent_eval.evaluation.ab_test import ABTestSummary
from agent_eval.evaluation.builtin import get_builtin_evaluator_instances
from agent_eval.evaluation.engine import EvaluationEngine
from agent_eval.report import terminal_report
from agent_eval.report.html_report import HTMLReportGenerator
from agent_eval.trace.models import RunRecord, RunStatus


def _evaluate_offline(run: RunRecord, spans) -> list:
    """Run the 5 built-in evaluators directly (no storage, no LLM)."""
    results: list = []
    for evaluator in get_builtin_evaluator_instances():
        results.extend(evaluator.evaluate(run, spans))
    return results


@pytest.fixture
def console_output(monkeypatch):
    """Replace the module-level rich Console with one writing to a wide StringIO.

    Wide width prevents rich from wrapping/clipping table cells in the captured text.
    """
    buf = io.StringIO()
    monkeypatch.setattr(terminal_report, "console", Console(file=buf, width=250))
    return buf


@pytest.fixture
def reporter(patch_config, tmp_path):
    """HTMLReportGenerator writing into a throwaway directory."""
    return HTMLReportGenerator(output_dir=tmp_path)


# ============================================================
# Terminal: single-run trace timeline
# ============================================================


class TestTerminalTraceTimeline:
    def test_shows_run_overview_and_spans(self, console_output, sample_run, sample_spans):
        terminal_report.print_trace_timeline(sample_run, sample_spans)
        out = console_output.getvalue()
        assert sample_run.run_id in out
        assert sample_run.task_id in out
        assert sample_run.agent_name in out
        assert "SUCCESS" in out
        assert "Step 0" in out
        assert "calculator" in out
        assert "gpt-4o-mini" in out
        assert "37" in out  # final answer panel

    def test_token_and_cost_annotations(self, console_output, sample_run, sample_spans):
        terminal_report.print_trace_timeline(sample_run, sample_spans)
        out = console_output.getvalue()
        assert "70tok" in out      # span_001 total tokens
        assert "$0.0001" in out    # span_001 cost

    def test_empty_spans_does_not_crash(self, console_output, sample_run):
        terminal_report.print_trace_timeline(sample_run, [])
        out = console_output.getvalue()
        assert "no spans recorded" in out


# ============================================================
# Terminal: single-run evaluation results
# ============================================================


class TestTerminalRunEvaluation:
    def test_prints_dimensions_and_scores(self, console_output, sample_run, sample_spans):
        results = _evaluate_offline(sample_run, sample_spans)
        terminal_report.print_run_evaluation(sample_run, results)
        out = console_output.getvalue()
        assert "Evaluation Results" in out
        for dim in ("success_rate", "tool_usage", "answer_quality", "latency", "token_cost"):
            assert dim in out
        assert "total=150" in out   # run token overview
        assert "5000" in out        # total latency
        # sample_run has both passing and failing sub-metrics
        assert "✅" in out and "❌" in out

    def test_empty_results_message(self, console_output, sample_run):
        terminal_report.print_run_evaluation(sample_run, [])
        out = console_output.getvalue()
        assert "No evaluation results available" in out


# ============================================================
# Terminal: full single-run report (timeline + evaluation)
# ============================================================


class TestFullSingleRunReport:
    def test_combines_timeline_and_evaluation(self, console_output, sample_run, sample_spans):
        results = _evaluate_offline(sample_run, sample_spans)
        terminal_report.print_full_single_run_report(sample_run, sample_spans, results)
        out = console_output.getvalue()
        assert "Agent Run Overview" in out
        assert "Execution Trace Timeline" in out
        assert "Evaluation Results" in out

    def test_none_results_skips_evaluation_section(self, console_output, sample_run, sample_spans):
        terminal_report.print_full_single_run_report(sample_run, sample_spans, None)
        out = console_output.getvalue()
        assert "Agent Run Overview" in out
        assert "Evaluation Results" not in out


# ============================================================
# Terminal: batch summary (real BatchSummary from the engine)
# ============================================================


class TestTerminalBatchSummary:
    def test_batch_summary_kpis_and_tables(self, console_output, storage, sample_run, sample_spans):
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)
        failed = RunRecord(
            run_id="fail_run",
            task_id="task_fail",
            status=RunStatus.FAILED,
            error_message="boom",
            input_text="This task failed",
            total_latency_ms=8000,
        )
        storage.save_run(failed)

        engine = EvaluationEngine(storage=storage)
        _, summary = engine.evaluate_runs(["test_run_001", "fail_run"])

        terminal_report.print_batch_summary(summary)
        out = console_output.getvalue()
        assert "Success Rate" in out
        assert "Quality Score" in out
        assert "Dimension-wise Statistics" in out
        assert "success_rate" in out
        assert "Top Failures" in out
        assert "fail_run" in out
        assert "50.0%" in out  # 1 of 2 runs succeeded
        assert "150" in out    # total tokens KPI (150 + 0)


# ============================================================
# Terminal: run list
# ============================================================


class TestTerminalRunList:
    def test_run_list_table(self, console_output, sample_run):
        terminal_report.print_run_list([sample_run])
        out = console_output.getvalue()
        assert "Recent Runs" in out
        assert sample_run.run_id in out
        assert sample_run.agent_name in out

    def test_empty_run_list(self, console_output):
        terminal_report.print_run_list([])
        out = console_output.getvalue()
        assert "No runs found" in out


# ============================================================
# HTML: single-run report
# ============================================================


class TestHTMLSingleRunReport:
    def test_generates_html_file(self, reporter, storage, sample_run, sample_spans):
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        path = reporter.generate_single_run_report("test_run_001", storage=storage)

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "<html" in html
        assert "chart.js" in html          # Chart.js CDN reference
        assert "test_run_001" in html
        assert "Execution Trace" in html
        assert "Evaluation Results" in html
        assert "badge-success" in html     # PASS marker
        assert "badge-fail" in html        # sample run also has failing sub-metrics

    def test_missing_run_raises_key_error(self, reporter, storage):
        with pytest.raises(KeyError):
            reporter.generate_single_run_report("ghost_run", storage=storage)

    def test_run_without_spans_and_output(self, reporter, storage):
        run = RunRecord(run_id="bare_run", status=RunStatus.SUCCESS, final_output=None,
                        input_text="empty task")
        storage.save_run(run)

        path = reporter.generate_single_run_report("bare_run", storage=storage)

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "bare_run" in html
        assert "(no output)" in html


# ============================================================
# HTML: batch summary report
# ============================================================


class TestHTMLBatchReport:
    def test_generates_batch_html(self, reporter, storage, sample_run, sample_spans):
        storage.save_run(sample_run)
        storage.append_spans(sample_spans)

        engine = EvaluationEngine(storage=storage)
        _, summary = engine.evaluate_runs(["test_run_001"])
        path = reporter.generate_batch_report(summary)

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "<html" in html
        assert "Batch Evaluation Report" in html
        assert "radarChart" in html and "<canvas" in html
        assert "Dimension Breakdown" in html
        assert "100.0%" in html  # success rate KPI for the single successful run


# ============================================================
# HTML: A/B comparison report
# ============================================================


class TestHTMLABReport:
    @staticmethod
    def _ab_summary() -> ABTestSummary:
        return ABTestSummary(
            agent_a_name="agent-a",
            agent_b_name="agent-b",
            dataset_name="toy",
            total_tasks=3,
            dimensions={
                "answer_quality": {
                    "agent_a": {"mean": 0.9, "pass_rate": 1.0},
                    "agent_b": {"mean": 0.5, "pass_rate": 0.6667},
                    "mean_delta": 0.4,
                    "pass_rate_delta": 0.3333,
                }
            },
            overall={
                "agent_a": {"success_rate": 1.0, "quality_score": 0.9,
                            "avg_latency_ms": 1000.0, "total_tokens": 330,
                            "total_cost": 0.0009},
                "agent_b": {"success_rate": 0.5, "quality_score": 0.5,
                            "avg_latency_ms": 5000.0, "total_tokens": 3000,
                            "total_cost": 0.003},
                "delta": {"success_rate": 0.5, "quality_score": 0.4,
                          "avg_latency_ratio": 0.2, "total_tokens": -2670,
                          "total_cost": -0.0021},
            },
            significance={
                "latency": {"mean_diff": -4000.0, "t_statistic": -34.641,
                            "significant": True, "n": 3},
                "tokens": {"mean_diff": -890.0, "t_statistic": -17.129,
                           "significant": True, "n": 3},
            },
            task_details=[],
        )

    def test_generates_ab_html(self, reporter):
        path = reporter.generate_ab_report(self._ab_summary())

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "<html" in html
        assert "A/B Comparison Report" in html
        assert "agent-a" in html and "agent-b" in html
        assert "abRadarChart" in html
        assert "Per-Dimension Comparison" in html
        assert "Significant" in html

    def test_empty_overall_data(self, reporter):
        summary = ABTestSummary(agent_a_name="a", agent_b_name="b",
                                dataset_name="d", total_tasks=0)
        path = reporter.generate_ab_report(summary)
        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "No comparison data available" in html
