"""ReAct (Reason + Act) agent implementation.

Uses function-calling when tools are available; otherwise falls back to
ReAct-style scratchpad prompting with Thought/Action/Action Input/Observation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent_eval.agent.base import BaseAgent, register_agent
from agent_eval.llm import (
    Message,
    Role,
    ToolCall,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from agent_eval.trace import RunRecord, RunStatus

logger = logging.getLogger("agent_eval.agent.react")


REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that solves tasks step by step using a ReAct (Reason + Act) loop.

You have access to a set of tools. On each step you should:
1. **Thought**: Briefly describe what you are thinking and why.
2. **Action**: Decide whether to use a tool OR give the final answer.
   - To use a tool: call the tool function with appropriate parameters.
   - To finish: respond with your final answer WITHOUT calling any tools.

Rules:
- Be concise in thoughts.
- Use tools whenever external information or computation is needed.
- If you are confident you can answer directly, do so as the final answer.
- If a tool returns unexpected results, adjust your reasoning and try again.
- Never output more than one Action per step.
"""


FALLBACK_REACT_PROMPT = """You must use the following EXACT format for each step:

```
Thought: <your reasoning here>
Action: <tool_name> OR Final Answer
Action Input: <JSON object of arguments, or final answer text>
```

AVAILABLE TOOLS (name + description):
{tools_list}

Begin!
"""


@register_agent
class ReActAgent(BaseAgent):
    """Standard ReAct agent with function-calling or scratchpad fallback."""

    agent_type: str = "react"

    def setup(self) -> None:
        self._use_function_calling = len(self.tools.list_tools()) > 0

    def run(self, task: str) -> tuple[str, RunRecord]:
        run = self._make_run(task)
        self.recorder.start_run(run)

        messages: list[Message] = self._build_initial_messages(task)

        final_answer: str | None = None
        try:
            for step in range(1, self.config.max_steps + 1):
                run.total_steps = step
                thought_text, message, is_final = self._reason_step(step, messages, run)

                if is_final or message.role == Role.ASSISTANT and not message.tool_calls:
                    # Either a forced final answer or a pure assistant message.
                    content = message.content or ""
                    if content.strip():
                        final_answer = content.strip()
                    if not final_answer:
                        final_answer = "(Agent returned empty final answer)"
                    messages.append(message)
                    self.recorder.on_step_end(run, step, "final_answer", final_answer)
                    break

                # Step has tool calls
                messages.append(message)
                action_desc = self._describe_action(message)
                observations: list[str] = []
                for tc in message.tool_calls or []:
                    tool_name = tc.name
                    arguments = tc.arguments or {}
                    result = self.tools.invoke(tool_name, arguments)
                    observation = result.output if result.success else f"[ERROR] {result.error}"
                    observations.append(f"[{tool_name}] {observation}")
                    messages.append(
                        tool_message(
                            content=observation,
                            tool_call_id=tc.id or f"call_{step}",
                            name=tool_name,
                        )
                    )
                self.recorder.on_step_end(run, step, action_desc, "\n".join(observations))
            else:
                # Loop exhausted without final answer
                if not final_answer:
                    final_answer = self._extract_last_content(messages) or (
                        "[Agent exceeded max steps without producing final answer]"
                    )
        except Exception as e:
            logger.exception(f"Agent run failed: {e}")
            self._finalize_run(
                run, output=final_answer, status=RunStatus.FAILED, error=f"{type(e).__name__}: {e}"
            )
            return (final_answer or f"Error: {e}", run)

        run = self._finalize_run(run, output=final_answer, status=RunStatus.SUCCESS)
        return (final_answer or "", run)

    # -------- Internal helpers --------

    def _build_initial_messages(self, task: str) -> list[Message]:
        messages: list[Message] = [system_message(REACT_SYSTEM_PROMPT)]
        if not self._use_function_calling:
            tools_list = "\n".join(
                f"- {t.name}: {t.description}" for t in self.tools.list_tools()
            )
            messages.append(system_message(FALLBACK_REACT_PROMPT.format(tools_list=tools_list)))
        messages.append(user_message(task))
        return messages

    def _reason_step(
        self, step: int, messages: list[Message], run: RunRecord
    ) -> tuple[str, Message, bool]:
        """Perform one LLM call; return (thought_text, message, is_final_answer)."""
        # 1) Emit a thought marker in trace BEFORE the LLM call
        thought_text = f"Step {step}: reasoning..."
        self.recorder.on_step_start(run, step, thought_text)

        try:
            tools_schema = self.tools.function_schemas() if self._use_function_calling else None
            response = self.llm.chat(
                messages,
                model=self.config.model,
                temperature=self.config.temperature,
                tools=tools_schema,
            )
        except Exception as e:
            logger.error(f"LLM call failed on step {step}: {e}")
            raise

        if self._use_function_calling and response.has_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.get("id"),
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", {}),
                )
                for tc in (response.tool_calls or [])
            ]
            thought = response.content or f"Calling tools: {[tc.name for tc in tool_calls]}"
            return thought, assistant_message(content=response.content, tool_calls=tool_calls), False

        # No tool calls: interpret as potential final answer
        content = response.content or ""
        parsed = self._try_parse_fallback_format(content)
        if parsed:
            action, action_input, thought = parsed
            if action.lower().startswith("final answer"):
                final_content = action_input.strip()
                return thought, assistant_message(content=final_content), True
            # Non-final Action in scratchpad mode -> synthesize function call
            args = action_input
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {"input": args}
            if not isinstance(args, dict):
                args = {"input": args}
            return (
                thought,
                assistant_message(
                    content=content,
                    tool_calls=[ToolCall(id=f"call_{step}", name=action, arguments=args)],
                ),
                False,
            )
        return content, assistant_message(content=content), True

    def _try_parse_fallback_format(
        self, content: str
    ) -> tuple[str, Any, str] | None:
        """Parse Thought/Action/Action Input from scratchpad format."""
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", content, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=\nAction Input:|\Z)", content, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.+)\Z", content, re.DOTALL)
        if action_match and input_match:
            thought = thought_match.group(1).strip() if thought_match else ""
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            return action, action_input, thought
        return None

    def _describe_action(self, message: Message) -> str:
        if message.tool_calls:
            parts = [f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})" for tc in message.tool_calls]
            return " | ".join(parts)
        return "final_answer"

    @staticmethod
    def _extract_last_content(messages: list[Message]) -> str | None:
        for m in reversed(messages):
            if m.role == Role.ASSISTANT and m.content:
                return m.content.strip()
        return None
