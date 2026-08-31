"""CLI tests using Typer's CliRunner.

All commands are exercised without any real LLM calls: storage directories are
redirected to a temp dir via env-var overrides (OUTPUT_DIR / RUN_DIR /
TRACE_DIR / EVAL_DIR, as configured in configs/default.yaml) and the config
cache is reset before/after every invocation so fresh env vars are picked up.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_eval import __version__
from agent_eval import config as config_mod
from agent_eval.cli import app
from agent_eval.trace.models import RunRecord, RunStatus, TokenUsage

runner = CliRunner()

ALL_COMMANDS = [
    "run",
    "eval",
    "report",
    "list",
    "evaluate",
    "compare",
    "migrate",
    "config",
    "judge",
    "serve",
    "errors",
    "compare-annotations",
]


@pytest.fixture
def cli_env(monkeypatch, tmp_output_dir):
    """Point config storage dirs at a temp dir and reset the config cache.

    configs/default.yaml reads ${OUTPUT_DIR} / ${TRACE_DIR} / ${RUN_DIR} /
    ${EVAL_DIR}, so setting these env vars + reset_config() before invoking
    makes the CLI (load_config / JSONLStorage) use the temp directories.
    reset_config() is called again on teardown (try/finally equivalent).
    """
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_output_dir))
    monkeypatch.setenv("TRACE_DIR", str(tmp_output_dir / "traces"))
    monkeypatch.setenv("RUN_DIR", str(tmp_output_dir / "runs"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_output_dir / "evaluations"))
    # Wide console so Rich tables are not truncated in captured output
    monkeypatch.setenv("COLUMNS", "200")
    config_mod.reset_config()
    yield tmp_output_dir
    config_mod.reset_config()


def _write_runs(run_dir: Path, runs: list[RunRecord]) -> None:
    """Write fake run records as JSONL, mimicking JSONLStorage.save_run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "runs.jsonl", "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r.to_storage_dict(), ensure_ascii=False) + "\n")


# ============================================================
# Global options (--version / --help)
# ============================================================


class TestGlobalOptions:
    def test_version_flag(self, cli_env):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"agent-eval v{__version__}" in result.output

    def test_version_short_flag(self, cli_env):
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "agent-eval v" in result.output

    def test_help_lists_all_commands(self, cli_env):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ALL_COMMANDS:
            assert cmd in result.output, f"command '{cmd}' missing from help output"


# ============================================================
# agent config
# ============================================================


class TestConfigCommand:
    def test_config_view_shows_current_configuration(self, cli_env):
        # No subcommand and no --set: defaults to show mode
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "Current Configuration" in result.output
        # The JSON dump of the config contains the main sections
        assert '"storage"' in result.output
        assert '"llm"' in result.output
        assert '"default_model"' in result.output
        assert '"pricing"' in result.output


# ============================================================
# agent list
# ============================================================


class TestListCommand:
    def test_list_shows_fake_runs(self, cli_env):
        runs = [
            RunRecord(
                run_id="cli_run_ok",
                task_id="task_a",
                agent_name="react",
                status=RunStatus.SUCCESS,
                input_text="What is 2+2?",
                final_output="4",
                total_steps=2,
                total_latency_ms=1500,
                tokens=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            ),
            RunRecord(
                run_id="cli_run_bad",
                task_id="task_b",
                agent_name="react",
                status=RunStatus.FAILED,
                input_text="What is 3+3?",
                error_message="APITimeoutError: request timed out",
            ),
        ]
        _write_runs(Path(cli_env) / "runs", runs)

        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Recent Runs" in result.output
        assert "cli_run_ok" in result.output
        assert "cli_run_bad" in result.output
        assert "task_a" in result.output
        assert "react" in result.output

    def test_list_empty_storage(self, cli_env):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No runs found" in result.output


# ============================================================
# agent errors
# ============================================================


class TestErrorsCommand:
    def test_errors_classifies_failed_runs(self, cli_env):
        runs = [
            RunRecord(
                run_id="err_run_timeout",
                task_id="task_err",
                agent_name="react",
                status=RunStatus.FAILED,
                input_text="timeout task",
                error_message="APITimeoutError: request timed out after 30s",
                total_steps=3,
                total_latency_ms=30000,
            ),
            RunRecord(
                run_id="err_run_ok",
                task_id="task_ok",
                agent_name="react",
                status=RunStatus.SUCCESS,
                input_text="fine task",
                final_output="ok",
            ),
        ]
        _write_runs(Path(cli_env) / "runs", runs)

        result = runner.invoke(app, ["errors"])
        assert result.exit_code == 0
        assert "Total Runs: 2" in result.output
        assert "Failure Rate" in result.output
        assert "Error Distribution" in result.output
        assert "LLM Timeout" in result.output
        assert "err_run_timeout" in result.output
        assert "Recent Failed Runs" in result.output

    def test_errors_no_failures(self, cli_env):
        runs = [
            RunRecord(
                run_id="ok_run_1",
                task_id="task_ok",
                agent_name="react",
                status=RunStatus.SUCCESS,
                input_text="fine task",
                final_output="ok",
            ),
        ]
        _write_runs(Path(cli_env) / "runs", runs)

        result = runner.invoke(app, ["errors"])
        assert result.exit_code == 0
        assert "Total Runs: 1" in result.output
        assert "no failed runs to classify" in result.output


# ============================================================
# agent run (error path only - no LLM calls)
# ============================================================


class TestRunCommand:
    def test_run_unknown_agent_type_errors(self, cli_env):
        result = runner.invoke(
            app, ["run", "Say hello", "--agent", "definitely_not_an_agent"]
        )
        assert result.exit_code != 0
        assert "Unknown agent type" in str(result.exception)
