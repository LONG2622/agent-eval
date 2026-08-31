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
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_eval import __version__
from agent_eval.config import load_config
from agent_eval.evaluation import ABTestRunner, EvaluationEngine, LLMJudgeEvaluator
from agent_eval.evaluation.token_efficiency import analyze_batch, analyze_run
from agent_eval.logger import setup_logger
from agent_eval.report import (
    HTMLReportGenerator,
    format_comparison_text,
    generate_comparison_report,
    print_batch_summary,
    print_full_single_run_report,
    print_run_list,
    run_comparison,
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
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Also log to a file (JSONL)."),
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
    model: str | None = typer.Option(None, "--model", "-m", help="Override LLM model name."),
    temperature: float | None = typer.Option(None, "--temperature", "-t", help="LLM temperature."),
    max_steps: int = typer.Option(10, "--max-steps", "-s", help="Max reasoning steps."),
    expected_output: str | None = typer.Option(None, "--expected", "-e", help="Expected answer for evaluation."),
    task_id: str | None = typer.Option(None, "--task-id", help="User-defined task identifier."),
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
    model: str | None = typer.Option(None, "--model", "-m", help="Override LLM model."),
    temperature: float | None = typer.Option(None, "--temperature", "-t", help="LLM temperature."),
    max_steps: int = typer.Option(10, "--max-steps", "-s", help="Max steps per task."),
    sample: int = typer.Option(0, "--sample", "-n", help="Randomly sample N tasks (0 = all)."),
    seed: int = typer.Option(42, "--seed", help="Seed for --sample."),
    workers: int = typer.Option(1, "--workers", "-w", help="Concurrent workers (phase 2)."),
    retries: int = typer.Option(2, "--retries", help="Max retry attempts per task."),
    retry_delay: float = typer.Option(2.0, "--retry-delay", help="Base retry delay (seconds)."),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Checkpoint file for resume."),
    report_html: bool = typer.Option(False, "--report-html", help="Generate HTML report after evaluation."),
) -> None:
    try:
        tasks = TaskDataset.from_jsonl(dataset)
    except (RuntimeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
        _err_console.print(f"[bold red]Failed to load dataset:[/] {e}")
        raise typer.Exit(code=1) from e
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
    task_id: str | None = typer.Option(None, "--task-id", help="Filter by task ID."),
) -> None:
    storage = JSONLStorage()
    runs = storage.list_runs(task_id=task_id)
    print_run_list(runs, limit=limit)


# ============================================================
# agent evaluate (re-evaluate without running)
# ============================================================


@app.command("evaluate", help="(Re-)evaluate existing run(s) by ID or all.")
def evaluate_cmd(
    run_ids: list[str] | None = typer.Argument(None, help="One or more run IDs. If omitted, evaluate ALL."),
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
    model_a: str | None = typer.Option(None, "--model-a", help="Model for agent A."),
    agent_b: str = typer.Option("react", "--agent-b", help="Agent type for B."),
    model_b: str | None = typer.Option(None, "--model-b", help="Model for agent B."),
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
    console.print("\n[bold green]=== A/B Comparison Result ===[/bold green]")
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
    db_path: Path | None = typer.Option(None, "--db", help="Target SQLite DB path (default: ./outputs/agent_eval.db)."),
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
    set_key: str | None = typer.Option(None, "--set", help="Set a config value (e.g. 'llm.temperature=0.5')."),
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
        with open(yaml_path, encoding="utf-8") as f:
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
        console.print("[bold cyan]Current Configuration:[/bold cyan]")
        console.print(json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False))


# ============================================================
# agent judge (LLM-as-Judge on existing runs)
# ============================================================

@app.command("judge", help="Run LLM-as-Judge evaluation on existing run(s).")
def judge_cmd(
    run_ids: list[str] | None = typer.Argument(None, help="Run ID(s) to judge. If omitted, judge ALL."),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Model to use as judge (default: current LLM)."),
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
            for _r in results:
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
        raise typer.Exit(code=1) from None


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


# ============================================================
# agent token-analysis (Token Efficiency Analysis)
# ============================================================

@app.command("token-analysis", help="Analyze token efficiency (redundancy, bloat, context usage).")
def token_analysis_cmd(
    run_id: str | None = typer.Option(None, "--run-id", help="Analyse a single run ID."),
    context_window: int | None = typer.Option(
        None, "--context-window", "-w", help="Override model context window (default: read from config)."
    ),
) -> None:
    storage = JSONLStorage()

    if run_id:
        run = storage.load_run(run_id)
        if run is None:
            _err_console.print(f"[bold red]Run ID not found:[/] {run_id}")
            raise typer.Exit(code=1)
        spans = storage.load_spans(run_id)
        analysis = analyze_run(run, spans, context_window=context_window)
        _print_single_run_token_analysis(analysis)
        return

    # Batch mode: analyse all runs
    batch = analyze_batch(storage, context_window=context_window)
    if not batch.runs:
        console.print("[yellow]No runs found in storage.[/]")
        raise typer.Exit(code=0)
    _print_batch_token_analysis(batch)


def _print_single_run_token_analysis(a) -> None:  # type: ignore[no-untyped-def]
    """Print a single RunTokenAnalysis using Rich."""
    from rich.table import Table

    table = Table(title=f"Token Efficiency — {a.run_id}", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Prompt tokens", f"{a.prompt_tokens:,}")
    table.add_row("Completion tokens", f"{a.completion_tokens:,}")
    table.add_row("Total tokens", f"{a.total_tokens:,}")
    table.add_row("Final answer chars", f"{a.final_answer_chars:,}")
    table.add_row("Chars / completion token", f"{a.chars_per_completion_token:.2f}")
    table.add_row("Duplicate prompt tokens", f"{a.duplicate_prompt_tokens:,}")
    table.add_row("Redundancy ratio", f"{a.redundancy_ratio:.1%}")
    table.add_row("Context window", f"{a.context_window:,}")
    table.add_row("Context used", f"{a.context_used_pct:.1f}%")
    table.add_row("System prompt tokens", f"{a.system_prompt_tokens:,}")
    table.add_row("Conversation bloat ratio", f"{a.conversation_bloat_ratio:.1%}")

    console.print(table)

    if a.flags:
        console.print("\n[bold yellow]⚠ Flags:[/bold yellow]")
        for flag in a.flags:
            console.print(f"  • [yellow]{flag}[/yellow]")
    else:
        console.print("\n[green]✔ No issues detected.[/green]")


def _print_batch_token_analysis(batch) -> None:  # type: ignore[no-untyped-def]
    """Print a BatchTokenAnalysis: per-run table + aggregate KPI card."""
    from rich.table import Table

    # Per-run table
    table = Table(title="Per-Run Token Efficiency", show_header=True, header_style="bold cyan")
    table.add_column("Run ID")
    table.add_column("Prompt", justify="right")
    table.add_column("Redundancy", justify="right")
    table.add_column("Ctx Used", justify="right")
    table.add_column("Chars/Token", justify="right")
    table.add_column("Dup Tokens", justify="right")
    table.add_column("Flags")

    for a in batch.runs:
        flags_str = ",".join(a.flags) if a.flags else "—"
        table.add_row(
            a.run_id[:16],
            f"{a.prompt_tokens:,}",
            f"{a.redundancy_ratio:.1%}",
            f"{a.context_used_pct:.1f}%",
            f"{a.chars_per_completion_token:.2f}",
            f"{a.duplicate_prompt_tokens:,}",
            flags_str,
        )
    console.print(table)

    # KPI card
    console.print("\n[bold green]=== Token Efficiency Summary ===[/bold green]")
    kpi = Table.grid(padding=(1, 2))
    kpi.add_column(style="dim")
    kpi.add_column(justify="right", style="bold")
    kpi.add_row("Runs analysed", f"{len(batch.runs)}")
    kpi.add_row("Avg redundancy", f"{batch.avg_redundancy_ratio:.1%}")
    kpi.add_row("Avg context used", f"{batch.avg_context_used_pct:.1f}%")
    kpi.add_row("Avg chars/token", f"{batch.avg_chars_per_token:.2f}")
    kpi.add_row("Total duplicate tokens", f"{batch.total_duplicate_tokens:,}")
    console.print(kpi)

    if batch.top_redundant_runs:
        console.print("\n[bold yellow]Top 5 Most Redundant Runs:[/bold yellow]")
        for rid, ratio in batch.top_redundant_runs[:5]:
            console.print(f"  • {rid[:16]} — {ratio:.1%}")

    if batch.recommendations:
        console.print("\n[bold cyan]💡 Recommendations:[/bold cyan]")
        for rec in batch.recommendations:
            console.print(f"  • {rec}")


# ============================================================
# agent baseline-save (save current batch as regression baseline)
# ============================================================

@app.command("baseline-save", help="Save current run(s) as a regression baseline.")
def baseline_save_cmd(
    name: str = typer.Option(..., "--name", "-n", help="Human-friendly baseline label (e.g. 'v1.0 release')."),
    run_ids: str | None = typer.Option(None, "--run-ids", help="Comma-separated run IDs. If omitted, save ALL runs."),
    dataset_id: str | None = typer.Option(None, "--dataset-id", help="Optional source dataset identifier."),
    agent_name: str | None = typer.Option(None, "--agent-name", help="Optional agent/model snapshot label."),
) -> None:
    from agent_eval.evaluation.baseline import save_baseline

    storage = JSONLStorage()
    engine = EvaluationEngine(storage=storage)

    resolved_ids: list[str] | None = None
    if run_ids:
        resolved_ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]

    try:
        with console.status("[cyan]Evaluating runs and saving baseline...[/cyan]"):
            baseline_id = save_baseline(
                engine=engine,
                run_ids=resolved_ids,
                name=name,
                dataset_id=dataset_id,
                agent_name=agent_name,
                storage=storage,
            )
    except ValueError as e:
        _err_console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(code=1) from e

    console.print(f"[green]✔ Baseline saved:[/] [bold]{baseline_id}[/]")
    console.print(f"  Name: {name}")
    console.print(f"  Runs: {len(resolved_ids) if resolved_ids else len(storage.list_runs())}")


# ============================================================
# agent baseline-compare (compare current runs vs saved baseline)
# ============================================================

@app.command("baseline-compare", help="Compare current runs against a saved regression baseline.")
def baseline_compare_cmd(
    baseline_id: str = typer.Option(..., "--baseline-id", "-b", help="Baseline ID to compare against."),
    run_ids: str | None = typer.Option(None, "--run-ids", help="Comma-separated current run IDs (default=ALL)."),
) -> None:
    from agent_eval.evaluation.baseline import compare_to_baseline, load_baseline

    storage = JSONLStorage()
    engine = EvaluationEngine(storage=storage)

    try:
        baseline = load_baseline(baseline_id)
    except FileNotFoundError as e:
        _err_console.print(f"[bold red]{e}[/bold red]")
        raise typer.Exit(code=1) from e

    resolved_ids: list[str] | None = None
    if run_ids:
        resolved_ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]

    with console.status("[cyan]Comparing runs against baseline...[/cyan]"):
        result = compare_to_baseline(
            baseline_id=baseline_id,
            current_run_ids=resolved_ids,
            engine=engine,
            storage=storage,
        )

    # --- Render verdict panel ---
    has_regression = bool(result["regressions"])
    verdict_title = "⚠ Regression Detected" if has_regression else "✓ No Regression"
    verdict_color = "red bold" if has_regression else "green bold"

    baseline_sr = baseline.summary.get("overall_success_rate", 0.0)
    current_sr = round(baseline_sr + result["overall_delta"], 4)

    panel_lines = [
        f"[dim]Baseline:[/] {baseline.name} ([cyan]{baseline.baseline_id}[/cyan])  ({result['baseline_run_count']} runs)",
        f"[dim]Current :[/] {result['current_run_count']} run(s)",
        f"[dim]Baseline success rate :[/] {baseline_sr:.2%}",
        f"[dim]Current  success rate :[/] {current_sr:.2%}",
        f"[dim]Overall delta          :[/] [{'red' if result['overall_delta'] < 0 else 'green'}]{result['overall_delta'] * 100:+.2f} pp[/]",
        "",
        result["note"],
    ]
    console.print(Panel("\n".join(panel_lines), title=f"[{verdict_color}]{verdict_title}[/{verdict_color}]", border_style="yellow"))

    # --- Dimension delta table ---
    deltas = result["dimension_deltas"]
    if deltas:
        table = Table(title="Per-dimension delta (current − baseline)")
        table.add_column("Dimension", style="cyan", no_wrap=True)
        table.add_column("Delta", justify="right", no_wrap=True)
        table.add_column("Status", no_wrap=True)

        for dim_name in sorted(deltas):
            d = deltas[dim_name]
            if d < -1e-6:
                status = "[red]regressed[/red]"
            elif d > 1e-6:
                status = "[green]improved[/green]"
            else:
                status = "[dim]unchanged[/dim]"
            table.add_row(
                dim_name,
                f"{d * 100:+.2f} pp",
                status,
            )
        console.print(table)
    else:
        console.print("[dim]No dimension-level data to compare.[/dim]")


# ============================================================
# agent baseline-list (list saved baselines)
# ============================================================

@app.command("baseline-list", help="List all saved regression baselines.")
def baseline_list_cmd() -> None:
    from agent_eval.evaluation.baseline import list_baselines

    baselines = list_baselines()
    if not baselines:
        console.print("[yellow]No baselines saved yet.[/yellow] Use `agent baseline-save` to create one.")
        return

    table = Table(title=f"Saved baselines ({len(baselines)})")
    table.add_column("Baseline ID", style="cyan", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Runs", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Created At", style="dim", no_wrap=True)

    for b in baselines:
        sr = b.summary.get("overall_success_rate", 0.0)
        table.add_row(
            b.baseline_id,
            b.name,
            str(len(b.run_ids)),
            f"{sr:.2%}",
            b.created_at,
        )
    console.print(table)


if __name__ == "__main__":
    app()
