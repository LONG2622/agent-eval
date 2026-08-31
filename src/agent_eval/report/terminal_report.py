"""Rich-based terminal visualization for runs, traces, and evaluations."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from agent_eval.evaluation import BatchSummary, EvaluationResult
from agent_eval.trace import RunRecord, Span, SpanType

console = Console()


# ============================================================
# Single Run Trace Replay
# ============================================================


def _span_type_style(span_type: SpanType) -> tuple[str, str]:
    """Return (icon, color) for a span type."""
    return {
        SpanType.AGENT_STEP: ("🔁", "cyan"),
        SpanType.LLM_CALL: ("🧠", "blue"),
        SpanType.TOOL_CALL: ("🛠️ ", "green"),
        SpanType.THOUGHT: ("💭", "magenta"),
    }.get(span_type, ("•", "white"))


def print_trace_timeline(run: RunRecord, spans: list[Span]) -> None:
    """Pretty-print a trace as a time-line tree to the terminal."""
    console.print()
    header = Table.grid(padding=(0, 2))
    header.add_column(justify="right", style="bold dim")
    header.add_column()
    header.add_row("Run ID:", f"[bold yellow]{run.run_id}[/]")
    header.add_row("Task ID:", f"[bold]{run.task_id}[/]" or "-")
    header.add_row("Agent:", f"[cyan]{run.agent_name}[/]")
    header.add_row("Status:", _status_text(run.status.value))
    header.add_row("Input:", run.input_text[:200] + ("..." if len(run.input_text) > 200 else ""))
    console.print(Panel(header, title="🧪 Agent Run Overview", border_style="blue"))

    # Build time-line tree grouped by step
    step_groups: dict[int, list[Span]] = {}
    for s in spans:
        step_groups.setdefault(s.step_index, []).append(s)

    tree = Tree("📜 [bold]Execution Trace Timeline", guide_style="dim")
    if not step_groups:
        tree.add("[dim]<no spans recorded>")
    for step_idx in sorted(step_groups.keys()):
        group = step_groups[step_idx]
        step_latency = sum(s.latency_ms for s in group)
        step_node = tree.add(
            f"[cyan bold]Step {step_idx}[/cyan bold]  [dim]({step_latency} ms)[/dim]"
        )
        for s in sorted(group, key=lambda x: x.created_at):
            icon, color = _span_type_style(s.span_type)
            title_parts = [
                f"[{color}]{icon} {s.span_type.value}[/]",
                f"[bold]{s.name}[/]" if s.name else "",
                f"[dim]{s.latency_ms}ms[/]",
            ]
            if s.tokens.total_tokens:
                title_parts.append(f"[dim]{s.tokens.total_tokens}tok[/dim]")
            if s.cost:
                title_parts.append(f"[dim]${s.cost:.4f}[/dim]")
            if not s.is_success:
                title_parts.append("[red]✗ FAILED[/red]")
            span_node = step_node.add("  ".join(p for p in title_parts if p))
            _attach_span_details(span_node, s)

    console.print(tree)

    # Final output
    if run.final_output:
        console.print(
            Panel(
                Text(run.final_output, style="green"),
                title="✅ Final Answer",
                border_style="green",
            )
        )
    if run.error_message:
        console.print(
            Panel(
                Text(run.error_message, style="red"),
                title="❌ Error",
                border_style="red",
            )
        )


def _attach_span_details(node: Tree, s: Span) -> None:
    # LLM inputs
    if s.span_type == SpanType.LLM_CALL:
        msgs = s.input_data.get("messages") or []
        last_user = next(
            (m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"),
            "",
        )
        if last_user:
            node.add(f"[dim]user prompt:[/dim] {str(last_user)[:200]}")
        content = s.output_data.get("content")
        if content:
            node.add(Text(f"assistant: {str(content)[:300]}", style="blue"))
        if s.output_data.get("tool_calls"):
            for tc in s.output_data["tool_calls"]:
                node.add(
                    f"[green]→ tool_call:[/green] [bold]{tc.get('name','')}[/bold]"
                    f" [dim](args={tc.get('arguments', {})})[/dim]"
                )
    elif s.span_type == SpanType.TOOL_CALL:
        args = s.input_data.get("arguments", {})
        node.add(f"[dim]args:[/dim] {args}")
        output = (s.output_data.get("output") or "")[:300]
        if output:
            node.add(Text(f"output: {output}", style="green dim"))
        if s.exception:
            node.add(Text(f"exception: {s.exception}", style="red"))
    elif s.span_type == SpanType.AGENT_STEP:
        thought = s.input_data.get("thought")
        if thought:
            node.add(Text(f"thought: {str(thought)[:300]}", style="magenta dim"))
        action = s.output_data.get("action")
        obs = s.output_data.get("observation")
        if action:
            node.add(f"[dim]action:[/dim] {action}")
        if obs:
            node.add(Text(f"observation: {str(obs)[:200]}", style="dim"))


def _status_text(status: str) -> str:
    mapping = {
        "success": "[bold green]✔ SUCCESS[/]",
        "failed": "[bold red]✘ FAILED[/]",
        "timeout": "[bold yellow]⏱ TIMEOUT[/]",
        "running": "[bold blue]▶ RUNNING[/]",
        "pending": "[dim]○ PENDING[/dim]",
    }
    return mapping.get(status, status)


# ============================================================
# Single Run Evaluation
# ============================================================


def print_run_evaluation(
    run: RunRecord,
    results: list[EvaluationResult],
) -> None:
    """Pretty-print evaluation results for a single run."""
    if not results:
        console.print("[yellow]No evaluation results available for this run.[/]")
        return

    # Overview card
    overview_tokens = run.tokens
    overview_grid = Table.grid(padding=(0, 2))
    overview_grid.add_column(justify="right", style="bold dim")
    overview_grid.add_column()
    overview_grid.add_row("Total Latency:", f"[bold]{run.total_latency_ms}[/] ms")
    overview_grid.add_row(
        "Tokens:",
        f"prompt={overview_tokens.prompt_tokens}, "
        f"completion={overview_tokens.completion_tokens}, "
        f"[bold]total={overview_tokens.total_tokens}[/]",
    )
    overview_grid.add_row("Total Cost:", f"[bold green]${run.total_cost:.5f}[/] USD")
    overview_grid.add_row("Total Steps:", f"{run.total_steps}")
    console.print(Panel(overview_grid, title="📊 Run Metrics", border_style="magenta"))

    # Results table
    table = Table(title="Evaluation Results", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Dimension", style="cyan", no_wrap=True)
    table.add_column("Sub-Metric", style="blue")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Passed", justify="center")
    table.add_column("Details", style="dim", max_width=80)

    dim_colors = {
        "success_rate": "green",
        "tool_usage": "blue",
        "answer_quality": "magenta",
        "latency": "yellow",
        "token_cost": "cyan",
    }
    for r in sorted(results, key=lambda x: (x.dimension.value, x.sub_metric.value if x.sub_metric else "")):
        color = dim_colors.get(r.dimension.value, "white")
        passed = "✅" if r.passed else ("❌" if r.passed is False else "—")
        details_str = _short_details(r.details)
        table.add_row(
            f"[{color}]{r.dimension.value}[/{color}]",
            r.sub_metric.value if r.sub_metric else "[dim](overall)[/dim]",
            f"{r.score:g}",
            passed,
            details_str,
        )
    console.print(table)


def _short_details(details: Any) -> str:
    if details is None:
        return ""
    if isinstance(details, dict | list):
        import json

        s = json.dumps(details, ensure_ascii=False)
    else:
        s = str(details)
    if len(s) > 120:
        s = s[:117] + "..."
    return s


# ============================================================
# Batch Summary
# ============================================================


def print_batch_summary(summary: BatchSummary) -> None:
    """Pretty-print a batch evaluation summary with KPI cards + dimension table."""
    console.print()
    # KPI row
    kpi_table = Table.grid(padding=(1, 6))
    kpi_table.add_column()
    kpi_table.add_column()
    kpi_table.add_column()
    kpi_table.add_column()
    kpi_table.add_column()
    kpi_table.add_row(
        _kpi_card("✔ Success Rate", f"{summary.overall_success_rate*100:.1f}%", "green"),
        _kpi_card("🎯 Quality Score", f"{summary.overall_quality_score:.2f}", "magenta"),
        _kpi_card("⏱ Avg Latency", f"{summary.avg_latency_ms:,.0f} ms", "yellow"),
        _kpi_card("💵 Avg Cost", f"${summary.avg_cost:.4f}", "cyan"),
        _kpi_card("📦 Runs / Tokens", f"{summary.evaluated_runs} runs / {summary.total_tokens:,} tok", "blue"),
    )
    console.print(kpi_table)
    console.print()

    # Dimension summary table
    dim_table = Table(
        title="Dimension-wise Statistics",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    dim_table.add_column("Dimension", style="cyan")
    dim_table.add_column("N", justify="right")
    dim_table.add_column("Pass Rate", justify="right", style="green")
    dim_table.add_column("Mean", justify="right")
    dim_table.add_column("Median", justify="right")
    dim_table.add_column("Min", justify="right", style="red")
    dim_table.add_column("Max", justify="right", style="green")
    for dim, ds in summary.dimension_summaries.items():
        dim_table.add_row(
            f"[bold]{dim}[/]",
            f"{ds.count}",
            f"{ds.pass_rate*100:.1f}%",
            f"{ds.mean_score:.3f}",
            f"{ds.median_score:.3f}",
            f"{ds.min_score:.3f}",
            f"{ds.max_score:.3f}",
        )
    console.print(dim_table)
    console.print(f"[dim]💡 Total cost: [bold]${summary.total_cost:.5f}[/bold] USD[/dim]")

    # Top failures / successes
    if summary.top_failures:
        fail_table = Table(
            title="⚠️ Top Failures (lowest score)",
            box=box.SIMPLE,
            header_style="bold red",
            show_lines=False,
        )
        fail_table.add_column("Rank", style="red", justify="right")
        fail_table.add_column("Run ID", style="yellow")
        fail_table.add_column("Task ID")
        fail_table.add_column("Score", justify="right")
        fail_table.add_column("Preview", style="dim", max_width=80)
        for i, item in enumerate(summary.top_failures, 1):
            fail_table.add_row(
                f"{i}", item["run_id"], item["task_id"], f"{item['score']:.3f}", item["preview"]
            )
        console.print(fail_table)

    if summary.top_successes:
        ok_table = Table(
            title="🌟 Top Successes (highest score)",
            box=box.SIMPLE,
            header_style="bold green",
        )
        ok_table.add_column("Rank", style="green", justify="right")
        ok_table.add_column("Run ID", style="yellow")
        ok_table.add_column("Task ID")
        ok_table.add_column("Score", justify="right")
        ok_table.add_column("Preview", style="dim", max_width=80)
        for i, item in enumerate(summary.top_successes, 1):
            ok_table.add_row(
                f"{i}", item["run_id"], item["task_id"], f"{item['score']:.3f}", item["preview"]
            )
        console.print(ok_table)


def _kpi_card(label: str, value: str, color: str) -> Panel:
    return Panel(
        Text.from_markup(f"[{color} bold]{value}[/{color} bold]\n[dim]{label}[/dim]"),
        box=box.ROUNDED,
        border_style=color,
        width=28,
    )


# ============================================================
# Run List
# ============================================================


def print_run_list(runs: list[RunRecord], limit: int = 50) -> None:
    if not runs:
        console.print("[yellow]No runs found yet. Use `agent run` first![/]")
        return
    table = Table(title=f"Recent Runs (showing {min(limit, len(runs))} of {len(runs)})", box=box.ROUNDED)
    table.add_column("Run ID", style="yellow")
    table.add_column("Task ID")
    table.add_column("Agent", style="cyan")
    table.add_column("Status")
    table.add_column("Steps", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Started", style="dim")
    for r in runs[:limit]:
        table.add_row(
            r.run_id,
            r.task_id or "-",
            r.agent_name,
            _status_text(r.status.value),
            f"{r.total_steps}",
            f"{r.total_latency_ms:,} ms",
            f"{r.tokens.total_tokens:,}",
            f"${r.total_cost:.4f}",
            r.started_at.replace("T", " ")[:19],
        )
    console.print(table)


# ============================================================
# Convenience: print full single-run report
# ============================================================


def print_full_single_run_report(
    run: RunRecord,
    spans: list[Span],
    eval_results: list[EvaluationResult] | None,
) -> None:
    print_trace_timeline(run, spans)
    if eval_results:
        print_run_evaluation(run, eval_results)
