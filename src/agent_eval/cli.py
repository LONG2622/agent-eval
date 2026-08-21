"""CLI entrypoint for agent-eval using Typer.

Commands:
    agent run          -- Run the agent on a single task.
    agent eval         -- Run + evaluate on a dataset (JSONL).
    agent report       -- Show trace + evaluation for a specific run ID.
    agent list         -- List recent runs.
    agent evaluate     -- Re-evaluate existing run(s) without re-running.
    agent compare      -- A/B test between two agent configurations.  [NEW]
    agent migrate      -- Migrate data between storage backends.       [NEW]
    agent config       -- View / modify configuration.                [NEW]
    agent judge        -- Run LLM-as-Judge on existing runs.          [NEW]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console

from agent_eval import __version__
from agent_eval.config import load_config
from agent_eval.evaluation import ABTestRunner, EvaluationEngine, LLMJudgeEvaluator
from agent_eval.logger import setup_logger
from agent_eval.report import (
    HTMLReportGenerator,
    format_comparison_text,
    generate_comparison_report,
    run_comparison,
    print_batch_summary,
    print_full_single_run_report,
    print_run_list,
)
from agent_eval.task import TaskDataset, TaskItem, TaskRunner
from agent_eval.trace import JSONLStorage, SQLiteStorage

app = typer.Typer(
    add_completion=False,
    help="Agent Eval - Run agents, record traces, evaluate systematically.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
_err_console = Console(stderr=True)
logger = setup_logger("agent_eval.cli")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"agent-eval v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Also log to a file (JSONL)."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON line logs on stdout."),
) -> None:
    setup_logger(
        level=10 if verbose else 20,
        log_file=log_file,
        json_output=json_logs,
    )
    load_config()  # ensure directories exist


# ============================================================
# agent run
# ============================================================


@app.command("run", help="Run a single task interactively and show trace + evaluation.")
def run_cmd(
    task: str = typer.Argument(..., help="The task description / user question."),
    agent_type: str = typer.Option("react", "--agent", "-a", help="Agent type (react, ...)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override LLM model name."),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="LLM temperature."),
    max_steps: int = typer.Option(10, "--max-steps", "-s", help="Max reasoning steps."),
    expected_output: Optional[str] = typer.Option(None, "--expected", "-e", help="Expected answer for evaluation."),
    task_id: Optional[str] = typer.Option(None, "--task-id", help="User-defined task identifier."),
) -> None:
    logger.info(f"Run started: agent={agent_type}, model={model or 'default'}, max_steps={max_steps}")
    runner = TaskRunner()
    task_item = TaskItem(
        input=task,
        expected_output=expected_output,
    )
    if task_id:
        task_item.task_id = task_id
    with console.status("[bold green]Agent is thinking...[/bold green]", spinner="dots"):
        outcome = runner.run_single(
            task_item,
            agent_type=agent_type,
            model=model,
            temperature=temperature,
            max_steps=max_steps,
            auto_evaluate=True,
        )
    spans = runner.storage.load_spans(outcome.run.run_id)
    eval_results = runner.evaluation_engine.get_run_results(outcome.run.run_id)
    print_full_single_run_report(outcome.run, spans, eval_results)
    _print_run_paths(outcome.run.run_id)


# ============================================================
# agent eval
# ============================================================


@app.command("eval", help="Run + evaluate an agent on a JSONL dataset.")
def eval_cmd(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to dataset JSONL."),
    agent_type: str = typer.Option("react", "--agent", "-a", help="Agent type."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override LLM model."),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="LLM temperature."),
    max_steps: int = typer.Option(10, "--max-steps", "-s", help="Max steps per task."),
    sample: int = typer.Option(0, "--sample", "-n", help="Randomly sample N tasks (0 = all)."),
    seed: int = typer.Option(42, "--seed", help="Seed for --sample."),
    workers: int = typer.Option(1, "--workers", "-w", help="Concurrent workers (phase 2)."),
    retries: int = typer.Option(2, "--retries", help="Max retry attempts per task."),
    retry_delay: float = typer.Option(2.0, "--retry-delay", help="Base retry delay (seconds)."),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", help="Checkpoint file for resume."),
    report_html: bool = typer.Option(False, "--report-html", help="Generate HTML report after evaluation."),
) -> None:
    try:
        tasks = TaskDataset.from_jsonl(dataset)
    except (RuntimeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
        _err_console.print(f"[bold red]Failed to load dataset:[/] {e}")
        raise typer.Exit(code=1)
    if sample > 0:
        tasks = tasks.sample(sample, seed=seed)
    if not tasks.items:
        _err_console.print("[yellow]Dataset is empty. Nothing to do.[/]")
        raise typer.Exit(code=0)

    console.print(
        f"[bold cyan]Loaded dataset '{tasks.name}' with {len(tasks)} tasks. "
        f"Agent={agent_type}, model={model or 'default'}, workers={workers}. Starting evaluation...[/]"
    )

    runner = TaskRunner()
    pbar = _ProgressTracker(total=len(tasks), console=console)

    def _progress(i: int, total: int, outcome) -> None:  # type: ignore[no-untyped-def]
        pbar.update(i, total, outcome)

    outcomes, summary = runner.run_batch(
        tasks,
        agent_type=agent_type,
        model=model,
        temperature=temperature,
        max_steps=max_steps,
        workers=workers,
        max_retries=retries,
        retry_delay=retry_delay,
        resume_from=checkpoint,
        progress_callback=_progress,
    )
    pbar.finish()

    if summary is None:
        _err_console.print("[red]Batch evaluation summary generation failed.[/]")
        raise typer.Exit(code=2)
    print_batch_summary(summary)
    logger.info(f"Evaluation completed: {len(outcomes)} runs, success_rate={summary.overall_success_rate:.4f}")

    # Generate HTML report if requested
    if report_html:
        gen = HTMLReportGenerator()
        with console.status("[dim]Generating HTML report...[/dim]"):
            path = gen.generate_batch_report(summary, label=tasks.name)
        console.print(f"[green]📊 HTML report saved:[/] {path}")

    # Print a few samples
    success_ids = [o.run.run_id for o in outcomes if o.run.status.value == "success"][:1]
    if success_ids:
        console.print(f"\n[dim]💡 Tip: inspect a run with:[/dim] `agent report {success_ids[0]}`")


# ============================================================
# agent report
# ============================================================


@app.command("report", help="Show full trace + evaluation for a run ID.")
def report_cmd(
    run_id: str = typer.Argument(..., help="Run ID (see `agent list`)."),
    no_trace: bool = typer.Option(False, "--no-trace", help="Skip timeline, show evaluation only."),
    fmt: str = typer.Option("terminal", "--format", "-f", help="Output format: terminal or html."),
) -> None:
    storage = JSONLStorage()
    run = storage.load_run(run_id)
    if run is None:
        _err_console.print(f"[bold red]Run ID not found:[/] {run_id}")
        _err_console.print("[dim]Hint: run `agent list` to see recent run IDs.[/]")
        raise typer.Exit(code=1)

    if fmt == "html":
        gen = HTMLReportGenerator()
        with console.status("[dim]Generating HTML report...[/dim]"):
            path = gen.generate_single_run_report(run_id, storage=storage)
        console.print(f"[green]📊 HTML report saved:[/] {path}")
        return

    spans = storage.load_spans(run_id)
    engine = EvaluationEngine(storage=storage)
    results = engine.get_run_results(run_id)
    if results is None:
        with console.status("[dim]Evaluating existing run...[/dim]"):
            results = engine.evaluate_run(run_id)
    if no_trace:
        spans = []
    print_full_single_run_report(run, spans, results)
    _print_run_paths(run_id)


# ============================================================
# agent list
# ============================================================


@app.command("list", help="List recent runs.")
def list_cmd(
    limit: int = typer.Option(50, "--limit", "-n", help="Max number of runs to show."),
    task_id: Optional[str] = typer.Option(None, "--task-id", help="Filter by task ID."),
) -> None:
    storage = JSONLStorage()
    runs = storage.list_runs(task_id=task_id)
    print_run_list(runs, limit=limit)


# ============================================================
# agent evaluate (re-evaluate without running)
# ============================================================


@app.command("evaluate", help="(Re-)evaluate existing run(s) by ID or all.")
def evaluate_cmd(
    run_ids: Optional[list[str]] = typer.Argument(None, help="One or more run IDs. If omitted, evaluate ALL."),
) -> None:
    storage = JSONLStorage()
    engine = EvaluationEngine(storage=storage)
    if not run_ids:
        runs = storage.list_runs()
        if not runs:
            _err_console.print("[yellow]No runs found to evaluate.[/]")
            raise typer.Exit(code=0)
        run_ids = [r.run_id for r in runs]

    with console.status(f"[cyan]Evaluating {len(run_ids)} run(s)...[/cyan]"):
        per_run, summary = engine.evaluate_runs(run_ids, save_summary=True)
    console.print(f"[green]✔ Evaluated {len(per_run)} runs, {summary.total_evaluation_results} individual scores.[/]")
    print_batch_summary(summary)


# ============================================================
# Helpers
# ============================================================


class _ProgressTracker:
    """Simple batch progress reporter using Rich status."""

    def __init__(self, total: int, console: Console) -> None:
        self.total = total
        self.console = console
        self._n_success = 0
        self._n_fail = 0

    def update(self, i: int, total: int, outcome) -> None:  # type: ignore[no-untyped-def]
        status = getattr(outcome.run.status, "value", "unknown")
        if status == "success":
            self._n_success += 1
        else:
            self._n_fail += 1
        pct = i / total * 100
        self.console.log(
            f"[{pct:5.1f}%] {i}/{total}  "
            f"[green]✔{self._n_success}[/green] / [red]✘{self._n_fail}[/red]  "
            f"task={outcome.task.task_id}  status=[bold]{status}[/bold]"
        )

    def finish(self) -> None:
        self.console.log(
            f"[bold green]Batch finished:[/] {self._n_success} success, "
            f"{self._n_fail} failed out of {self.total}"
        )


def _print_run_paths(run_id: str) -> None:
    cfg = load_config().storage
    trace_file = Path(cfg.trace_dir) / f"{run_id}.jsonl"
    eval_file = Path(cfg.eval_dir) / f"eval_{run_id}.jsonl"
    lines = []
    if trace_file.exists():
        lines.append(f"  📜 trace  = [dim]{trace_file}[/dim]")
    if eval_file.exists():
        lines.append(f"  📊 eval   = [dim]{eval_file}[/dim]")
    if lines:
        rprint("\n[dim]Persistence files:[/dim]")
        rprint("\n".join(lines))


# ============================================================
# agent compare (A/B test)
# ============================================================

@app.command("compare", help="A/B test: compare two agent configurations on a dataset.")
def compare_cmd(
    dataset: Path = typer.Argument(..., exists=True, dir_okay=False, help="Path to dataset JSONL."),
    agent_a: str = typer.Option("react", "--agent-a", help="Agent type for A."),
    model_a: Optional[str] = typer.Option(None, "--model-a", help="Model for agent A."),
    agent_b: str = typer.Option("react", "--agent-b", help="Agent type for B."),
    model_b: Optional[str] = typer.Option(None, "--model-b", help="Model for agent B."),
    sample: int = typer.Option(0, "--sample", "-n", help="Randomly sample N tasks."),
    report_html: bool = typer.Option(False, "--report-html", help="Generate HTML comparison report."),
) -> None:
    tasks = TaskDataset.from_jsonl(dataset)
    if sample > 0:
        tasks = tasks.sample(sample)
    if not tasks.items:
        _err_console.print("[yellow]Dataset is empty.[/]")
        raise typer.Exit(code=0)

    console.print(f"[cyan]A/B Test: {len(tasks)} tasks[/cyan]")

    runner = ABTestRunner()
    summary = runner.compare(
        tasks,
        agent_a={"agent_type": agent_a, "model": model_a},
        agent_b={"agent_type": agent_b, "model": model_b},
        label_a=f"A-{model_a or agent_a}",
        label_b=f"B-{model_b or agent_b}",
    )

    # Print summary
    console.print(f"\n[bold green]=== A/B Comparison Result ===[/bold green]")
    console.print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))

    if report_html:
        gen = HTMLReportGenerator()
        path = gen.generate_ab_report(summary)
        console.print(f"\n[green]📊 A/B report saved:[/] {path}")


# ============================================================
# agent migrate
# ============================================================

@app.command("migrate", help="Migrate data between storage backends (JSONL → SQLite).")
def migrate_cmd(
    source_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Source directory containing runs/ and traces/."),
    db_path: Optional[Path] = typer.Option(None, "--db", help="Target SQLite DB path (default: ./outputs/agent_eval.db)."),
) -> None:
    trace_dir = source_dir / "traces"
    run_dir = source_dir / "runs"
    if not run_dir.exists():
        _err_console.print(f"[bold red]No 'runs/' directory found in {source_dir}[/]")
        raise typer.Exit(code=1)

    storage = SQLiteStorage(db_path=db_path)
    with console.status("[cyan]Migrating JSONL → SQLite...[/cyan]"):
        counts = storage.migrate_from_jsonl(trace_dir, run_dir)

    console.print(f"[green]✔ Migration complete: {counts['runs']} runs, {counts['spans']} spans imported.[/]")
    agg = storage.query_aggregates()
    if agg:
        console.print(f"[dim]Aggregates: {json.dumps(agg, indent=2)}[/dim]")


# ============================================================
# agent config
# ============================================================

@app.command("config", help="View or modify the current configuration.")
def config_cmd(
    show: bool = typer.Option(True, "--show/--no-show", help="Display current config."),
    set_key: Optional[str] = typer.Option(None, "--set", help="Set a config value (e.g. 'llm.temperature=0.5')."),
) -> None:
    cfg = load_config()

    if set_key:
        if "=" not in set_key:
            _err_console.print("[red]Format: --set 'key.path=value'[/]")
            raise typer.Exit(code=1)
        key_path, value = set_key.split("=", 1)
        # Reload config from YAML file, modify, and write back
        import yaml
        yaml_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Navigate into nested dict
        parts = key_path.split(".")
        d = raw
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        # Try to coerce value
        try:
            coerced = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            coerced = value
        d[parts[-1]] = coerced
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        console.print(f"[green]✔ Set {key_path} = {coerced}[/green]")
        # Reset config cache
        import agent_eval.config as cfg_mod
        cfg_mod._config_instance = None
        return

    if show:
        console.print(f"[bold cyan]Current Configuration:[/bold cyan]")
        console.print(json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False))


# ============================================================
# agent judge (LLM-as-Judge on existing runs)
# ============================================================

@app.command("judge", help="Run LLM-as-Judge evaluation on existing run(s).")
def judge_cmd(
    run_ids: Optional[list[str]] = typer.Argument(None, help="Run ID(s) to judge. If omitted, judge ALL."),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Model to use as judge (default: current LLM)."),
) -> None:
    storage = JSONLStorage()
    if not run_ids:
        runs = storage.list_runs()
        if not runs:
            _err_console.print("[yellow]No runs found to judge.[/]")
            raise typer.Exit(code=0)
        run_ids = [r.run_id for r in runs]

    judge = LLMJudgeEvaluator(judge_model=judge_model)
    engine = EvaluationEngine(storage=storage)

    console.print(f"[cyan]Running LLM-as-Judge on {len(run_ids)} run(s)...[/cyan]")
    for rid in run_ids:
        with console.status(f"[dim]Judging {rid[:12]}...[/dim]"):
            run = storage.load_run(rid)
            if run is None:
                console.print(f"[yellow]  Skip {rid[:12]}: not found[/yellow]")
                continue
            spans = storage.load_spans(rid)
            results = judge.evaluate(run, spans)
            # Persist judge results
            for r in results:
                engine._persist_run_results(rid, results)
        console.print(f"[green]  ✔ Judged {rid[:12]}: {len(results)} scores[/green]")

    console.print(f"[bold green]✔ LLM-as-Judge complete: {len(run_ids)} runs evaluated.[/]")


# ============================================================
# agent serve (Web Dashboard + API Server)
# ============================================================


@app.command("serve", help="Start the web dashboard + REST API server.")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload on code changes."),
    open_browser: bool = typer.Option(True, "--no-open/--open", help="Auto-open browser."),
) -> None:
    import subprocess
    import sys
    import webbrowser

    from agent_eval import __version__

    console.print(
        f"[bold cyan]Starting Agent Eval Server v{__version__}[/bold cyan]"
    )
    console.print(f"  📡 API:      http://{host}:{port}/api")
    console.print(f"  📊 Dashboard: http://{host}:{port}/")
    console.print(f"  🔄 Reload:   {'on' if reload else 'off'}")
    console.print()
    logger.info(f"Server started: host={host}, port={port}, reload={reload}")

    if open_browser:
        import threading
        def _open_browser():
            import time
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}/")
        threading.Thread(target=_open_browser, daemon=True).start()

    # Run uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "agent_eval.server.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except ImportError:
        _err_console.print("[red]uvicorn not installed. Run: pip install uvicorn[standard][/]")
        raise typer.Exit(code=1)


# ============================================================
# agent errors (Error Classification)
# ============================================================

@app.command("errors", help="Classify and summarize failed runs by error type.")
def errors_cmd(
    limit: int = typer.Option(0, "--limit", "-n", help="Max recent errors to show (0=all)."),
) -> None:
    from agent_eval.evaluation.error_classifier import classify_all_runs, format_summary_text

    storage = JSONLStorage()
    summary = classify_all_runs(storage, limit=limit)
    console.print(format_summary_text(summary))


# ============================================================
# agent compare-annotations (Human vs Auto Eval Comparison)
# ============================================================

@app.command("compare-annotations", help="Compare human annotations with automatic evaluation scores.")
def compare_annotations_cmd(
    save_report: bool = typer.Option(False, "--save", help="Save comparison report as JSON file."),
) -> None:
    storage = JSONLStorage()

    with console.status("[cyan]Running annotation vs auto-evaluation comparison...[/cyan]"):
        summary, items = run_comparison(storage)

    console.print(format_comparison_text(summary, items))

    if save_report:
        path = generate_comparison_report(storage)
        console.print(f"\n[green]📊 Comparison report saved:[/] {path}")


if __name__ == "__main__":
    app()
