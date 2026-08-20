"""Tests for ToolRegistry, FunctionTool, @tool decorator, and builtin tools."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_eval.tools.builtin import calculator, get_current_time, read_file, web_search
from agent_eval.tools.registry import (
    FunctionTool,
    ToolCallContext,
    ToolRegistry,
    ToolResult,
    tool,
)


# ============================================================
# @tool Decorator Tests
# ============================================================


class TestToolDecorator:
    def test_basic_decorator(self):
        """@tool should convert function to FunctionTool."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert isinstance(add, FunctionTool)
        assert add.name == "add"
        assert add.description == "Add two numbers."

    def test_decorator_with_name(self):
        """@tool(name=...) should override function name."""

        @tool(name="my_add")
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        assert add.name == "my_add"

    def test_decorator_with_description(self):
        """@tool(description=...) should override docstring."""

        @tool(description="Custom description")
        def add(a: int, b: int) -> int:
            """Original docstring."""
            return a + b

        assert add.description == "Custom description"

    def test_tool_run_success(self):
        """FunctionTool.run() should execute the function and return ToolResult."""

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        result = add.run(a=3, b=5)
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "8"  # json.dumps of int
        assert result.latency_ms >= 0

    def test_tool_run_invalid_args(self):
        """Invalid args should produce a failed ToolResult."""

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        result = add.run(a="not_an_int", b=5)
        assert result.success is False
        assert "Invalid arguments" in result.error

    def test_tool_run_exception(self):
        """Exceptions in function should be caught and returned as failed result."""

        @tool
        def fail_tool() -> str:
            raise RuntimeError("Something went wrong")

        result = fail_tool.run()
        assert result.success is False
        assert "RuntimeError" in result.error
        assert "Something went wrong" in result.error

    def test_tool_optional_param(self):
        """Parameters with defaults should be optional."""

        @tool
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        # With default
        r1 = greet.run(name="World")
        assert r1.success is True
        assert "Hello, World!" in r1.output

        # With explicit
        r2 = greet.run(name="World", greeting="Hi")
        assert r2.success is True
        assert "Hi, World!" in r2.output

    def test_tool_to_function_schema(self):
        """to_function_schema should return OpenAI-compatible schema."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers.
            Args:
                a: First number
                b: Second number
            """
            return a + b

        schema = add.to_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["description"]
        assert "properties" in schema["function"]["parameters"]
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "b" in schema["function"]["parameters"]["properties"]

    def test_tool_validate_args_success(self):
        """validate_args should work for valid args."""

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        validated = add.validate_args({"a": 3, "b": 5})
        assert validated == {"a": 3, "b": 5}

    def test_tool_validate_args_failure(self):
        """validate_args should raise ValidationError for invalid args."""

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        with pytest.raises(ValidationError):
            add.validate_args({"a": "bad", "b": 5})


# ============================================================
# ToolRegistry Tests
# ============================================================


class TestToolRegistry:
    def test_register_function_tool(self):
        """Register a function-based tool."""
        reg = ToolRegistry()

        @tool
        def my_tool(x: int) -> str:
            return str(x)

        reg.register(my_tool)
        assert reg.get("my_tool") is my_tool

    def test_register_raw_callable(self):
        """Register a raw callable should auto-wrap it."""
        reg = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        tool_obj = reg.register(add)
        assert isinstance(tool_obj, FunctionTool)
        assert tool_obj.name == "add"

    def test_register_duplicate_raises(self):
        """Registering duplicate tool should raise ValueError."""
        reg = ToolRegistry()

        @tool
        def dup() -> str:
            return "dup"

        reg.register(dup)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(dup)

    def test_unregister(self):
        """unregister should remove tool."""
        reg = ToolRegistry()

        @tool
        def temp() -> str:
            return "temp"

        reg.register(temp)
        reg.unregister("temp")
        with pytest.raises(KeyError):
            reg.get("temp")

    def test_get_unknown_raises(self):
        """Getting unknown tool should raise KeyError."""
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_tools(self):
        """list_tools should return all registered tools."""
        reg = ToolRegistry()

        @tool
        def t1() -> str:
            return "1"

        @tool
        def t2() -> str:
            return "2"

        reg.register(t1)
        reg.register(t2)
        tools = reg.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"t1", "t2"}

    def test_function_schemas(self):
        """function_schemas should return list of schemas."""
        reg = ToolRegistry()

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        reg.register(add)
        schemas = reg.function_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "add"

    def test_invoke_success(self):
        """invoke should execute tool and return result."""
        reg = ToolRegistry()

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        reg.register(add)
        result = reg.invoke("add", {"a": 3, "b": 5})
        assert result.success is True
        assert "8" in result.output

    def test_invoke_unknown_tool(self):
        """Invoking unknown tool should raise KeyError."""
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.invoke("nonexistent", {})

    def test_callbacks(self):
        """Callbacks should be triggered on tool invocation."""
        reg = ToolRegistry()

        @tool
        def add(a: int, b: int) -> int:
            return a + b

        reg.register(add)

        events = []

        class TestCallback:
            def on_tool_start(self, ctx: ToolCallContext):
                events.append(("start", ctx.tool_name))

            def on_tool_end(self, ctx: ToolCallContext):
                events.append(("end", ctx.tool_name))

            def on_tool_error(self, ctx: ToolCallContext):
                events.append(("error", ctx.tool_name))

        cb = TestCallback()
        reg.register_callback(cb)
        reg.invoke("add", {"a": 1, "b": 2})

        assert ("start", "add") in events
        assert ("end", "add") in events
        assert ("error", "add") not in events

    def test_callback_on_error(self):
        """Error callback should fire when a raw tool raises an exception."""
        from agent_eval.tools.registry import BaseTool, ToolResult

        class FailTool(BaseTool):
            name = "fail_tool"
            description = "A tool that always fails"

            def run(self, **kwargs) -> ToolResult:
                raise RuntimeError("Intentional failure")

        reg = ToolRegistry()
        reg.register(FailTool())

        events = []

        class TestCallback:
            def on_tool_start(self, ctx: ToolCallContext):
                pass

            def on_tool_end(self, ctx: ToolCallContext):
                pass

            def on_tool_error(self, ctx: ToolCallContext):
                events.append(ctx.error)

        reg.register_callback(TestCallback())
        result = reg.invoke("fail_tool", {})
        assert result.success is False
        assert len(events) == 1
        assert isinstance(events[0], RuntimeError)


# ============================================================
# Builtin Tools Tests
# ============================================================


class TestBuiltinTools:
    def test_calculator_simple(self):
        """Calculator should evaluate simple expressions."""
        result = calculator.run(expression="2 + 3 * 4")
        assert result.success is True
        assert "14" in result.output

    def test_calculator_complex(self):
        """Calculator should handle complex expressions."""
        result = calculator.run(expression="sqrt(144) + 5^2")
        assert result.success is True
        assert "37" in result.output

    def test_calculator_invalid(self):
        """Calculator should return error for invalid expressions."""
        result = calculator.run(expression="invalid ++++")
        assert result.success is False

    def test_get_current_time(self):
        """get_current_time should return a valid time string."""
        result = get_current_time.run()
        assert result.success is True
        assert result.output  # non-empty

    def test_read_file(self, tmp_output_dir):
        """read_file should read file contents."""
        test_file = tmp_output_dir / "test.txt"
        test_file.write_text("Hello World!")
        result = read_file.run(file_path=str(test_file))
        assert result.success is True
        assert "Hello World!" in result.output

    def test_read_file_not_found(self):
        """read_file should fail for nonexistent files."""
        result = read_file.run(file_path="/nonexistent/file.txt")
        assert result.success is False

    def test_web_search(self):
        """web_search should return results (or handle gracefully)."""
        result = web_search.run(query="Python programming")
        # May fail due to network, but should return a ToolResult
        assert isinstance(result, ToolResult)
        # Should have output or error
        assert result.output or result.error