"""Message types for LLM interactions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Represents a tool call request from the LLM."""

    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str | None = None
    name: str | None = None  # For role=tool, the tool name
    tool_call_id: str | None = None  # For role=tool, maps to ToolCall.id
    tool_calls: list[ToolCall] | None = None  # For role=assistant

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dict format."""
        d: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            import json as _json

            d["tool_calls"] = [
                {
                    "id": tc.id or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json.dumps(tc.arguments, ensure_ascii=False)
                        if isinstance(tc.arguments, dict)
                        else str(tc.arguments),
                    },
                }
                for i, tc in enumerate(self.tool_calls)
            ]
        return d


def system_message(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def user_message(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def assistant_message(
    content: str | None = None, tool_calls: list[ToolCall] | None = None
) -> Message:
    return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)


def tool_message(content: str, tool_call_id: str, name: str | None = None) -> Message:
    return Message(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)
