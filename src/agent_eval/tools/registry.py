"""Tool data models and registry."""

from __future__ import annotations

import inspect
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError, create_model

from agent_eval.logger import setup_logger

logger = setup_logger("agent_eval.tools.registry")


# -------------------- Tool Parameter Schemas --------------------


def _type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python type annotation to OpenAI function-calling JSON schema."""
    if annotation is str or annotation == "str":
        return {"type": "string"}
    if annotation is int or annotation == "int":
        return {"type": "integer"}
    if annotation is float or annotation == "float":
        return {"type": "number"}
    if annotation is bool or annotation == "bool":
        return {"type": "boolean"}
    if annotation is list or annotation == "list":
        return {"type": "array", "items": {}}
    if annotation is dict or annotation == "dict":
        return {"type": "object"}
    # Default: treat as string
    return {"type": "string"}


def _docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from Google/NumPy-style docstrings."""
    if not docstring:
        return {}
    descriptions: dict[str, str] = {}
    # Match patterns like "param_name: description" or "Args: section"
    lines = docstring.splitlines()
    in_args_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "parameters:", "arguments:"):
            in_args_section = True
            continue
        if in_args_section and stripped and not stripped.startswith(" "):
            # Probably a new section
            if re.match(r"^[A-Z][a-z]+:", stripped):
                in_args_section = False
                continue
        if in_args_section:
            m = re.match(r"^\s*(\w+)\s*[:\-\(]\s*(.+)$", line)
            if m:
                descriptions[m.group(1)] = m.group(2).strip()
    return descriptions


# -------------------- Tool Wrapper --------------------


@dataclass
class ToolResult:
    """Standardized result of a tool invocation."""

    output: str
    success: bool = True
    error: str | None = None
    latency_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallContext:
    """Context for tool call callbacks (for tracing)."""

    tool_name: str
    arguments: dict[str, Any]
    started_at: float = field(default_factory=time.time)
    result: ToolResult | None = None
    error: Exception | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ToolCallback(ABC):  # noqa: B024 - marker base with optional hooks
    """Hook for tracing tool invocations."""

    def on_tool_start(self, ctx: ToolCallContext) -> None:  # noqa: B027
        pass

    def on_tool_end(self, ctx: ToolCallContext) -> None:  # noqa: B027
        pass

    def on_tool_error(self, ctx: ToolCallContext) -> None:  # noqa: B027
        pass


# -------------------- Base Tool Abstraction --------------------


class BaseTool(ABC):
    """Abstract base class for a tool."""

    name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError

    def to_function_schema(self) -> dict[str, Any]:
        """Return OpenAI-compatible tool function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters_schema.get("properties", {}),
                    "required": self.parameters_schema.get("required", []),
                },
            },
        }


# -------------------- Function-Based Tool --------------------


class FunctionTool(BaseTool):
    """Wraps a Python function into a BaseTool via the @tool decorator."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._func = func
        self.name = name or func.__name__
        sig = inspect.signature(func)
        doc = description or func.__doc__ or ""
        self.description = doc.split("\n\n")[0].strip() or f"Tool: {self.name}"
        param_descriptions = _docstring_params(doc)

        properties: dict[str, Any] = {}
        required: list[str] = []
        fields: dict[str, Any] = {}

        for p_name, p in sig.parameters.items():
            if p_name in ("self", "cls"):
                continue
            ann = Any if p.annotation is inspect.Parameter.empty else p.annotation
            prop = _type_to_json_schema(ann)
            if p_name in param_descriptions:
                prop["description"] = param_descriptions[p_name]
            properties[p_name] = prop
            if p.default is inspect.Parameter.empty:
                required.append(p_name)
                fields[p_name] = (ann, ...)
            else:
                fields[p_name] = (ann, p.default)

        self.parameters_schema = {
            "properties": properties,
            "required": required,
        }
        self._validator_model = create_model(
            f"{self.name}_args",
            __config__=None,
            **fields,  # type: ignore[arg-type]
        )

    def validate_args(self, raw_args: dict[str, Any]) -> dict[str, Any]:
        """Validate raw arguments with Pydantic. Raises ValidationError on failure."""
        model_instance = self._validator_model.model_validate(raw_args)
        return model_instance.model_dump()

    def run(self, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            validated = self.validate_args(kwargs)
        except ValidationError as e:
            return ToolResult(
                output="",
                success=False,
                error=f"Invalid arguments: {e}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        try:
            result_value = self._func(**validated)
            if isinstance(result_value, ToolResult):
                result_value.latency_ms = int((time.perf_counter() - started) * 1000)
                return result_value
            if isinstance(result_value, str):
                output = result_value
            else:
                try:
                    output = json.dumps(result_value, ensure_ascii=False, indent=2)
                except (TypeError, ValueError):
                    output = str(result_value)
            return ToolResult(
                output=output,
                success=True,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            return ToolResult(
                output="",
                success=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


# -------------------- Tool Decorator --------------------


def tool(name: str | None = None, description: str | None = None) -> Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator to turn a Python function into a FunctionTool.

    Usage:
        @tool
        def add(a: int, b: int) -> int:
            '''Add two numbers.
            Args:
                a: First number
                b: Second number
            '''
            return a + b

        # Or with explicit name:
        @tool(name="addition")
        def add(a: int, b: int) -> int: ...
    """

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)

    # Support both @tool and @tool(name="x")
    if callable(name) and description is None:
        # User wrote @tool without parentheses
        actual_func = name
        return FunctionTool(actual_func)
    return decorator


# -------------------- Tool Registry --------------------


class ToolRegistry:
    """Central registry for tools used by agents."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._callbacks: list[ToolCallback] = []

    def register(self, tool_obj: BaseTool | Callable[..., Any]) -> BaseTool:
        """Register a tool (either a BaseTool instance or a raw callable)."""
        if isinstance(tool_obj, BaseTool):
            pass
        elif callable(tool_obj):
            tool_obj = FunctionTool(tool_obj)
        else:
            raise TypeError(f"Cannot register type {type(tool_obj)} as tool")
        if tool_obj.name in self._tools:
            raise ValueError(f"Tool '{tool_obj.name}' already registered")
        self._tools[tool_obj.name] = tool_obj
        return tool_obj

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}. Available: {sorted(self._tools)}")
        return self._tools[name]

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def function_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible list of tool schemas."""
        return [t.to_function_schema() for t in self._tools.values()]

    # ---- Callbacks ----

    def register_callback(self, cb: ToolCallback) -> None:
        self._callbacks.append(cb)

    def unregister_callback(self, cb: ToolCallback) -> None:
        if cb in self._callbacks:
            self._callbacks.remove(cb)

    # ---- Execution ----

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a tool by name with validated arguments, triggering callbacks."""
        tool_instance = self.get(tool_name)
        ctx = ToolCallContext(tool_name=tool_name, arguments=dict(arguments))
        for cb in self._callbacks:
            cb.on_tool_start(ctx)
        try:
            result = tool_instance.run(**arguments)
            ctx.result = result
            for cb in self._callbacks:
                cb.on_tool_end(ctx)
            return result
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            ctx.error = e
            for cb in self._callbacks:
                cb.on_tool_error(ctx)
            return ToolResult(
                output="",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )
