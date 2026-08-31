"""Base agent abstractions and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agent_eval.llm import LLMGateway
from agent_eval.tools import ToolRegistry
from agent_eval.trace import RunRecord, RunStatus, TraceRecorder


class AgentRunConfig(BaseModel):
    """Config for one agent execution."""

    agent_name: str = "react-default"
    agent_type: str = "react"
    model: str | None = None
    temperature: float | None = None
    max_steps: int = 10
    step_timeout_seconds: int = 60
    task_id: str = ""
    expected_output: str | None = None
    ground_truth: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Lifecycle:
        1. setup()   - initialize state (llm gateway, tools, callbacks, recorder)
        2. run(task) - execute the Reason+Act loop, returning final answer
        3. cleanup() - optional teardown
    """

    agent_type: str = "base"

    def __init__(
        self,
        llm_gateway: LLMGateway,
        tool_registry: ToolRegistry,
        recorder: TraceRecorder,
        config: AgentRunConfig | None = None,
    ) -> None:
        self.llm = llm_gateway
        self.tools = tool_registry
        self.recorder = recorder
        self.config = config or AgentRunConfig()
        # Hook recorder into LLM + tools (idempotent: skip if already registered)
        if recorder not in self.llm._callbacks:
            self.llm.register_callback(recorder)
        if recorder not in self.tools._callbacks:
            self.tools.register_callback(recorder)
        self.setup()

    # ---- Extension points ----

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Initialize per-agent state (memory, prompt templates, etc.)."""
        pass

    @abstractmethod
    def run(self, task: str) -> tuple[str, RunRecord]:
        """Execute the agent's reasoning loop.

        Returns:
            (final_answer_text, run_record)
        """
        raise NotImplementedError

    def cleanup(self) -> None:  # noqa: B027 - optional hook
        """Release resources if needed."""
        pass

    # ---- Helpers ----

    def _make_run(self, task: str) -> RunRecord:
        return RunRecord(
            task_id=self.config.task_id or "",
            agent_name=self.config.agent_name,
            agent_config={
                "agent_type": self.agent_type,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_steps": self.config.max_steps,
            },
            input_text=task,
            expected_output=self.config.expected_output,
            ground_truth=self.config.ground_truth,
            metadata=self.config.metadata,
        )

    def _finalize_run(
        self,
        run: RunRecord,
        *,
        output: str | None,
        status: RunStatus,
        error: str | None = None,
    ) -> RunRecord:
        return self.recorder.end_run(run, status=status, final_output=output, error=error)


# -------------------- Agent Registry --------------------


class AgentRegistry:
    """Discover and instantiate agent classes by type string."""

    def __init__(self) -> None:
        self._classes: dict[str, type[BaseAgent]] = {}

    def register(self, agent_class: type[BaseAgent]) -> None:
        self._classes[agent_class.agent_type] = agent_class

    def get_class(self, agent_type: str) -> type[BaseAgent]:
        if agent_type not in self._classes:
            raise KeyError(
                f"Unknown agent type '{agent_type}'. "
                f"Available: {sorted(self._classes)}"
            )
        return self._classes[agent_type]

    def create(
        self,
        agent_type: str,
        llm_gateway: LLMGateway,
        tool_registry: ToolRegistry,
        recorder: TraceRecorder,
        config: AgentRunConfig | None = None,
    ) -> BaseAgent:
        cls = self.get_class(agent_type)
        return cls(llm_gateway, tool_registry, recorder, config)


_AGENT_REGISTRY = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    return _AGENT_REGISTRY


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Class decorator: register an agent class globally."""
    _AGENT_REGISTRY.register(cls)
    return cls
