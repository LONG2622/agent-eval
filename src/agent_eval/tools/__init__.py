"""Tools package."""

from agent_eval.tools.builtin import (
    calculator,
    get_current_time,
    read_file,
    register_builtin_tools,
    web_search,
)
from agent_eval.tools.registry import (
    BaseTool,
    FunctionTool,
    ToolCallback,
    ToolCallContext,
    ToolRegistry,
    ToolResult,
    tool,
)

__all__ = [
    "BaseTool",
    "FunctionTool",
    "ToolCallback",
    "ToolCallContext",
    "ToolRegistry",
    "ToolResult",
    "tool",
    "calculator",
    "web_search",
    "get_current_time",
    "read_file",
    "register_builtin_tools",
]
