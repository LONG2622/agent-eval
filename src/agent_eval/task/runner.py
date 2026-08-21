"""Task dataset model + Task Runner for batch execution.

v2 enhancements (Phase 2):
  - Concurrent execution via thread pool (workers > 1)
  - Automatic retry with exponential backoff on failure
  - Checkpoint / resume support via a JSONL progress file
  - Rate limiting (simple token-bucket style)
  - Rich progress reporting with ETA
"""

from __future__ import annotations

import json
import threading
from agent_eval.logger import setup_logger
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

from pydantic import BaseModel, Field

from agent_eval.agent import AgentRunConfig, BaseAgent, get_agent_registry
from agent_eval.config import load_config
from agent_eval.llm import LLMGateway
from agent_eval.tools import ToolRegistry, register_builtin_tools
from agent_eval.trace import JSONLStorage, RunRecord, TraceRecorder

if TYPE_CHECKING:
    from agent_eval.evaluation import BatchSummary, EvaluationEngine

logger = setup_logger("agent_eval.task.runner")


# ============================================================
# Task Dataset Model
# ============================================================


class TaskItem(BaseModel):
    """A single task definition from a dataset."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    input: str
    expected_output: str | None = None
    ground_truth: dict[str, Any] | None = None
    eval_criteria: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskItem":
        # Support alternative common keys (question/answer/prompt)
        input_text = (
            data.get("input")
            or data.get("question")
            or data.get("prompt")
            or data.get("query")
            or ""
        )
        expected = data.get("expected_output") or data.get("answer") or data.get("reference")
        task_id = data.get("task_id") or data.get("id") or uuid.uuid4().hex[:10]
        return cls(
            task_id=str(task_id),
            input=str(input_text),
            expected_output=str(expected) if expected else None,
            ground_truth=data.get("ground_truth"),
            eval_criteria=data.get("eval_criteria"),
            metadata=data.get("metadata", {}),
        )


class TaskDataset:
    """Collection of TaskItems loaded from JSONL/CSV or list."""

    def __init__(self, items: list[TaskItem] | None = None, name: str = "default") -> None:
        self.items: list[TaskItem] = items or []
        self.name = name

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[TaskItem]:
        return iter(self.items)

    # ---- Loaders ----

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> "TaskDataset":
        dataset_name = name or Path(path).stem
        items: list[TaskItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    items.append(TaskItem.from_dict(data))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid line {i} in {path}: {e}")
        return cls(items=items, name=dataset_name)

    @classmethod
    def from_list(cls, items: Iterable[dict[str, Any]], name: str = "in_memory") -> "TaskDataset":
        task_items = [TaskItem.from_dict(d) for d in items]
        return cls(items=task_items, name=name)

    # ---- Utility ----

    def sample(self, n: int, seed: int | None = 42) -> "TaskDataset":
        import random

        rng = random.Random(seed)
        sampled = rng.sample(self.items, min(n, len(self.items)))
        return TaskDataset(items=sampled, name=f"{self.name}_n{n}")


# ============================================================
# Task Runner
# ============================================================


@dataclass
class RunOutcome:
    task: TaskItem
    output: str
    run: RunRecord


# -------- Rate limiter --------

@dataclass
class _RateLimiter:
    """Simple token-bucket rate limiter for API calls."""
    max_per_second: float = 5.0
    _tokens: float = field(init=False, default=5.0)
    _last_time: float = field(init=False, default_factory=time.time)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def acquire(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_time
            self._tokens += elapsed * self.max_per_second
            if self._tokens > self.max_per_second:
                self._tokens = self.max_per_second
            self._last_time = now
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self.max_per_second
                time.sleep(wait_time)
                now = time.time()
                elapsed = now - self._last_time
                self._tokens += elapsed * self.max_per_second
                self._last_time = now
            self._tokens -= 1.0


class TaskRunner:
    """High-level runner: create common deps, run agents, trigger evaluation.

    v2 enhancements: concurrency, retry, checkpoint/resume, rate limiting.
    """

    def __init__(
        self,
        llm_gateway: LLMGateway | None = None,
        tool_registry: ToolRegistry | None = None,
        trace_recorder: TraceRecorder | None = None,
        storage: JSONLStorage | None = None,
        evaluation_engine: "EvaluationEngine" | None = None,
    ) -> None:
        from agent_eval.evaluation import EvaluationEngine  # noqa: F811

        cfg = load_config()
        self.cfg = cfg
        self.storage = storage or JSONLStorage()
        self.llm = llm_gateway or LLMGateway()
        self.tools = tool_registry or ToolRegistry()
        register_builtin_tools(self.tools)
        self.recorder = trace_recorder or TraceRecorder(self.storage)
        self.llm.register_callback(self.recorder)
        self.tools.register_callback(self.recorder)
        self.evaluation_engine = evaluation_engine or EvaluationEngine(storage=self.storage)
        self.agent_registry = get_agent_registry()
        self._rate_limiter = _RateLimiter()

    # -------- Single task --------

    def run_single(
        self,
        task: TaskItem | str,
        *,
        agent_type: str | None = None,
        agent_name: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_steps: int | None = None,
        auto_evaluate: bool = True,
    ) -> RunOutcome:
        if isinstance(task, str):
            task = TaskItem(input=task)
        cfg = AgentRunConfig(
            agent_name=agent_name or f"{agent_type or self.cfg.agent.default_type}-{task.task_id[:6]}",
            agent_type=agent_type or self.cfg.agent.default_type,
            model=model,
            temperature=temperature,
            max_steps=max_steps or self.cfg.agent.max_steps,
            step_timeout_seconds=self.cfg.agent.step_timeout_seconds,
            task_id=task.task_id,
            expected_output=task.expected_output,
            ground_truth=task.ground_truth,
            metadata=task.metadata,
        )
        agent = self.agent_registry.create(
            cfg.agent_type, self.llm, self.tools, self.recorder, cfg
        )
        try:
            output, run = agent.run(task.input)
        finally:
            agent.cleanup()
        if auto_evaluate:
            try:
                self.evaluation_engine.evaluate_run(run.run_id)
            except (RuntimeError, ValueError, ConnectionError, TimeoutError, OSError, KeyError, AttributeError) as e:
                logger.warning(f"Auto-evaluation failed for run {run.run_id}: {e}")
        return RunOutcome(task=task, output=output, run=run)

    # -------- Batch (enhanced v2) --------

    def run_batch(
        self,
        dataset: TaskDataset,
        *,
        agent_type: str | None = None,
        agent_name_prefix: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_steps: int | None = None,
        workers: int = 1,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        resume_from: str | Path | None = None,
        progress_callback: Callable[[int, int, RunOutcome | None], None] | None = None,
    ) -> tuple[list[RunOutcome], "BatchSummary | None"]:
        """Run a batch of tasks with concurrency, retry, and checkpoint support.

        Args:
            dataset: TaskDataset to execute.
            workers: Number of concurrent threads (1 = sequential).
            max_retries: Max retry attempts per task on failure.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            resume_from: Path to a checkpoint JSONL file to resume from.
            progress_callback: Optional callback(completed, total, latest_outcome).

        Returns:
            (list of RunOutcome, BatchSummary or None)
        """
        outcomes: list[RunOutcome] = []
        total = len(dataset)
        completed_ids: set[str] = set()

        # Resume from checkpoint
        if resume_from:
            completed_ids = self._load_checkpoint(resume_from)
            logger.info(f"Resuming: {len(completed_ids)} tasks already done")

        if workers <= 1:
            outcomes = self._run_sequential(
                dataset, agent_type, agent_name_prefix, model, temperature,
                max_steps, max_retries, retry_delay, completed_ids,
                progress_callback, resume_from,
            )
        else:
            outcomes = self._run_concurrent(
                dataset, agent_type, agent_name_prefix, model, temperature,
                max_steps, workers, max_retries, retry_delay, completed_ids,
                progress_callback, resume_from,
            )

        # Aggregate evaluation summary
        run_ids = [o.run.run_id for o in outcomes]
        try:
            _, summary = self.evaluation_engine.evaluate_runs(run_ids, save_summary=True)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning(f"Batch aggregation failed: {e}")
            return outcomes, None
        return outcomes, summary

    # ---- Sequential runner ----

    def _run_sequential(
        self,
        dataset: TaskDataset,
        agent_type: str | None,
        agent_name_prefix: str,
        model: str | None,
        temperature: float | None,
        max_steps: int | None,
        max_retries: int,
        retry_delay: float,
        completed_ids: set[str],
        progress_callback: Callable | None,
        checkpoint_path: str | Path | None,
    ) -> list[RunOutcome]:
        outcomes: list[RunOutcome] = []
        completed = len(completed_ids)
        total = len(dataset)

        for task in dataset:
            if task.task_id in completed_ids:
                logger.debug(f"Skipping completed task {task.task_id}")
                continue
            logger.info(f"[{completed + 1}/{total}] Running task {task.task_id}: {task.input[:60]}...")
            outcome = self._run_single_with_retry(
                task, agent_type, agent_name_prefix, model, temperature,
                max_steps, max_retries, retry_delay,
            )
            outcomes.append(outcome)
            completed += 1
            completed_ids.add(task.task_id)
            if checkpoint_path:
                self._save_checkpoint(checkpoint_path, task.task_id, outcome)
            if progress_callback:
                progress_callback(completed, total, outcome)

        return outcomes

    # ---- Concurrent runner ----

    def _run_concurrent(
        self,
        dataset: TaskDataset,
        agent_type: str | None,
        agent_name_prefix: str,
        model: str | None,
        temperature: float | None,
        max_steps: int | None,
        workers: int,
        max_retries: int,
        retry_delay: float,
        completed_ids: set[str],
        progress_callback: Callable | None,
        checkpoint_path: str | Path | None,
    ) -> list[RunOutcome]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        outcomes: list[RunOutcome] = []
        outcomes_lock = threading.Lock()
        completed_count = len(completed_ids)
        count_lock = threading.Lock()
        total = len(dataset)

        pending_tasks = [t for t in dataset if t.task_id not in completed_ids]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task: dict = {}
            for task in pending_tasks:
                future = executor.submit(
                    self._run_single_with_retry,
                    task, agent_type, agent_name_prefix, model, temperature,
                    max_steps, max_retries, retry_delay,
                )
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    outcome = future.result()
                except (RuntimeError, ValueError, ConnectionError, TimeoutError, OSError, KeyError, AttributeError) as e:
                    logger.error(f"Task {task.task_id} thread failed: {e}")
                    outcome = RunOutcome(
                        task=task, output="",
                        run=self._make_failed_run(task, e),
                    )
                with outcomes_lock:
                    outcomes.append(outcome)
                with count_lock:
                    completed_count += 1
                    completed_ids.add(task.task_id)
                    if checkpoint_path:
                        self._save_checkpoint(checkpoint_path, task.task_id, outcome)
                if progress_callback:
                    progress_callback(completed_count, total, outcome)

        # Sort outcomes by task order in dataset
        task_order = {t.task_id: i for i, t in enumerate(dataset)}
        outcomes.sort(key=lambda o: task_order.get(o.task.task_id, 9999))
        return outcomes

    # ---- Retry logic ----

    def _run_single_with_retry(
        self,
        task: TaskItem,
        agent_type: str | None,
        agent_name_prefix: str,
        model: str | None,
        temperature: float | None,
        max_steps: int | None,
        max_retries: int,
        retry_delay: float,
    ) -> RunOutcome:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                self._rate_limiter.acquire()
                effective_type = agent_type or self.cfg.agent.default_type
                return self.run_single(
                    task,
                    agent_type=effective_type,
                    agent_name=f"{agent_name_prefix}{effective_type}-{task.task_id[:8]}",
                    model=model,
                    temperature=temperature,
                    max_steps=max_steps,
                    auto_evaluate=True,
                )
            except (RuntimeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Task {task.task_id} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Task {task.task_id} failed after {max_retries + 1} attempts: {e}")
        return RunOutcome(
            task=task, output="",
            run=self._make_failed_run(task, last_error or RuntimeError("Unknown error")),
        )

    # ---- Checkpoint helpers ----

    @staticmethod
    def _load_checkpoint(path: str | Path) -> set[str]:
        path = Path(path)
        completed: set[str] = set()
        if not path.exists():
            return completed
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("status") == "success":
                        completed.add(data["task_id"])
                except (OSError, IOError, TypeError, ValueError):
                    pass
        return completed

    @staticmethod
    def _save_checkpoint(path: str | Path, task_id: str, outcome: RunOutcome) -> None:
        path = Path(path)
        status = "success" if outcome.run.status == "success" else "failed"
        entry = {
            "task_id": task_id,
            "run_id": outcome.run.run_id,
            "status": status,
            "timestamp": time.time(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -------- Helpers --------

    def _make_failed_run(self, task: TaskItem, error: Exception) -> RunRecord:
        cfg = self.cfg
        run = RunRecord(
            task_id=task.task_id,
            agent_name=f"{cfg.agent.default_type}-{task.task_id[:8]}",
            agent_config={
                "agent_type": cfg.agent.default_type,
                "model": cfg.llm.default_model,
                "max_steps": cfg.agent.max_steps,
            },
            input_text=task.input,
            expected_output=task.expected_output,
        )
        from agent_eval.trace import RunStatus

        self.recorder.start_run(run)
        # Record the error as a span so latency is tracked
        from agent_eval.trace import Span, SpanType

        error_span = Span(
            trace_id=run.trace_id,
            span_type=SpanType.AGENT_STEP,
            step_index=1,
            name="error",
            output_data={"error": str(error)},
            is_success=False,
            exception=f"{type(error).__name__}: {error}",
        )
        self.recorder._add_span(error_span)
        return self.recorder.end_run(run, status=RunStatus.FAILED, error=f"{type(error).__name__}: {error}")
