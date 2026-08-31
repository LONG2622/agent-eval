"""Tests for TaskRunner using a stub agent (no LLM, zero network calls).

The runner is exercised end-to-end: dataset loading, batch execution with
retry / checkpoint / concurrency, and BatchSummary aggregation.
"""

from __future__ import annotations

import json

import pytest

from agent_eval.agent import get_agent_registry
from agent_eval.agent.base import BaseAgent
from agent_eval.task import TaskDataset, TaskRunner
from agent_eval.trace import RunStatus

# -------------------- Stub agent --------------------


class StubAgent(BaseAgent):
    """Deterministic agent: returns ("stub answer", success RunRecord).

    Class-level shared state, because the runner constructs a fresh agent
    instance for every task. ``fail_times`` makes the first N total calls
    raise RuntimeError to exercise the retry path.
    """

    agent_type = "stub"

    state = {"fail_times": 0, "attempts": 0, "successes": 0}

    def run(self, task: str):
        cls = type(self)
        cls.state["attempts"] += 1
        if cls.state["attempts"] <= cls.state["fail_times"]:
            raise RuntimeError("stub deliberate failure")
        run = self._make_run(task)
        self.recorder.start_run(run)
        run = self._finalize_run(run, output="stub answer", status=RunStatus.SUCCESS)
        cls.state["successes"] += 1
        return ("stub answer", run)


get_agent_registry().register(StubAgent)


def _reset_stub(fail_times: int = 0) -> None:
    StubAgent.state.update({"fail_times": fail_times, "attempts": 0, "successes": 0})


def _write_dataset(path, tasks: list[dict]) -> TaskDataset:
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return TaskDataset.from_jsonl(path)


@pytest.fixture
def runner(patch_config):
    """TaskRunner on tmp storage via the patched config (no ./outputs pollution)."""
    from agent_eval.evaluation import EvaluationEngine
    from agent_eval.trace import JSONLStorage

    storage = JSONLStorage()
    engine = EvaluationEngine(storage=storage)
    _reset_stub()
    return TaskRunner(storage=storage, evaluation_engine=engine)


_THREE_TASKS = [
    {"task_id": "t1", "input": "2+2=?", "expected_output": "stub answer"},
    {"task_id": "t2", "input": "3+3=?", "expected_output": "stub answer"},
    {"task_id": "t3", "input": "4+4=?", "expected_output": "stub answer"},
]


# -------------------- Basic batch --------------------


def test_run_batch_creates_and_saves_three_successful_runs(runner, tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.jsonl", _THREE_TASKS)

    outcomes, summary = runner.run_batch(dataset, agent_type="stub", workers=1)

    assert len(outcomes) == 3
    assert StubAgent.state["attempts"] == 3
    assert [o.task.task_id for o in outcomes] == ["t1", "t2", "t3"]
    for outcome in outcomes:
        assert outcome.run.status == RunStatus.SUCCESS
        assert outcome.output == "stub answer"
        assert outcome.run.error_message is None

    # All 3 runs were persisted to storage.
    saved = runner.storage.list_runs()
    assert len(saved) == 3
    assert {r.status for r in saved} == {RunStatus.SUCCESS}

    # Batch summary counts.
    assert summary is not None
    assert summary.total_runs == 3
    assert summary.evaluated_runs == 3
    assert summary.overall_success_rate == 1.0


def test_run_single_with_string_task(runner):
    outcome = runner.run_single("hello stub", agent_type="stub")

    assert outcome.run.status == RunStatus.SUCCESS
    assert outcome.output == "stub answer"
    assert outcome.run.input_text == "hello stub"
    assert runner.storage.load_run(outcome.run.run_id) is not None


# -------------------- Retry --------------------


def test_retry_recovers_from_transient_failure(runner, tmp_path):
    _reset_stub(fail_times=1)  # first call fails, second succeeds
    dataset = _write_dataset(
        tmp_path / "one.jsonl", [{"task_id": "r1", "input": "hi", "expected_output": "stub answer"}]
    )

    outcomes, summary = runner.run_batch(
        dataset, agent_type="stub", workers=1, max_retries=2, retry_delay=0.0
    )

    assert StubAgent.state["attempts"] == 2  # 1 failed attempt + 1 retry
    assert len(outcomes) == 1
    assert outcomes[0].run.status == RunStatus.SUCCESS
    assert outcomes[0].output == "stub answer"
    assert summary.total_runs == 1
    assert summary.overall_success_rate == 1.0


def test_retry_exhaustion_produces_failed_run(runner, tmp_path):
    _reset_stub(fail_times=5)  # more failures than attempts available
    dataset = _write_dataset(
        tmp_path / "one.jsonl", [{"task_id": "x1", "input": "hi", "expected_output": "stub answer"}]
    )

    outcomes, _summary = runner.run_batch(
        dataset, agent_type="stub", workers=1, max_retries=2, retry_delay=0.0
    )

    assert StubAgent.state["attempts"] == 3  # initial attempt + 2 retries
    assert len(outcomes) == 1
    run = outcomes[0].run
    assert run.status == RunStatus.FAILED
    assert "RuntimeError" in run.error_message
    assert "stub deliberate failure" in run.error_message

    saved = runner.storage.list_runs()
    assert len(saved) == 1
    assert saved[0].status == RunStatus.FAILED


# -------------------- Checkpoint / resume --------------------


def test_checkpoint_resume_skips_completed_tasks(runner, tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    dataset = _write_dataset(
        tmp_path / "ds.jsonl",
        [
            {"task_id": "c1", "input": "a", "expected_output": "stub answer"},
            {"task_id": "c2", "input": "b", "expected_output": "stub answer"},
        ],
    )

    outcomes, _ = runner.run_batch(dataset, agent_type="stub", workers=1, resume_from=checkpoint)
    assert len(outcomes) == 2
    assert StubAgent.state["successes"] == 2

    # Checkpoint file written: one entry per completed task.
    lines = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {e["task_id"] for e in lines} == {"c1", "c2"}
    assert all(e["status"] == "success" for e in lines)
    assert all(e.get("run_id") for e in lines)

    # Resuming skips everything: no stub invocations, no new outcomes.
    _reset_stub()
    outcomes2, _ = runner.run_batch(dataset, agent_type="stub", workers=1, resume_from=checkpoint)
    assert outcomes2 == []
    assert StubAgent.state["attempts"] == 0


# -------------------- Concurrency --------------------


def test_workers_two_concurrent_batch_all_success(runner, tmp_path):
    dataset = _write_dataset(tmp_path / "dataset.jsonl", _THREE_TASKS)

    outcomes, summary = runner.run_batch(dataset, agent_type="stub", workers=2)

    assert StubAgent.state["attempts"] == 3
    assert len(outcomes) == 3
    assert all(o.run.status == RunStatus.SUCCESS for o in outcomes)
    assert all(o.output == "stub answer" for o in outcomes)
    # Concurrent outcomes are re-sorted back into dataset order.
    assert [o.task.task_id for o in outcomes] == ["t1", "t2", "t3"]

    assert len(runner.storage.list_runs()) == 3
    assert summary is not None
    assert summary.total_runs == 3
    assert summary.overall_success_rate == 1.0


# -------------------- Dataset loading errors --------------------


def test_malformed_jsonl_lines_are_skipped(runner, tmp_path):
    path = tmp_path / "broken.jsonl"
    good1 = json.dumps({"task_id": "m1", "input": "a", "expected_output": "stub answer"})
    good2 = json.dumps({"task_id": "m2", "input": "b", "expected_output": "stub answer"})
    path.write_text(
        good1 + "\nthis is {{{ not json\n" + good2 + "\n",
        encoding="utf-8",
    )

    # Source behaviour: invalid lines are skipped with a warning, not fatal.
    dataset = TaskDataset.from_jsonl(path)
    assert len(dataset) == 2
    assert [t.task_id for t in dataset] == ["m1", "m2"]

    outcomes, summary = runner.run_batch(dataset, agent_type="stub", workers=1)
    assert StubAgent.state["attempts"] == 2
    assert len(outcomes) == 2
    assert all(o.run.status == RunStatus.SUCCESS for o in outcomes)
    assert summary.total_runs == 2
