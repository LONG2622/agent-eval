"""ReAct (Reason + Act) agent implementation.

Uses function-calling when tools are available; otherwise falls back to
ReAct-style scratchpad prompting with Thought/Action/Action Input/Observation.
"""

from __future__ import annotations

import json
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
from agent_eval.logger import setup_logger
from agent_eval.trace import RunRecord, RunStatus

logger = setup_logger("agent_eval.agent.react")


REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that solves tasks step by step using a ReAct (Reason + Act) loop.

You have access to a set of tools. On each step you should:
1. **Thought**: Briefly describe what you are thinking and why.
2. **Action**: Decide whether to use a tool OR give the final answer.
   - To use a tool: call the tool function with appropriate parameters.
   - To finish: respond with your final answer WITHOUT calling any tools.

CRITICAL RULES:
- Respond in the SAME LANGUAGE as the user's question.
- Your final answer MUST be a direct answer to the user. NEVER include your internal reasoning, "Thought:", "Action:", or any step-by-step thinking in the final answer.
- After receiving tool results, you MUST try to produce a final answer. Only use more tools if you genuinely need more information.
- MAXIMUM 3 tool calls total. After 3 tool calls, you MUST give a final answer using all available information.
- Be concise in thoughts.
- Use tools whenever external information or computation is needed.
- If you are confident you can answer directly, do so as the final answer.
- If a tool returns unexpected results, adjust your reasoning and try again.
- Never output more than one Action per step.
- When you have enough information to answer, ALWAYS give the final answer instead of calling more tools.
- If asked what model you are, honestly state your model name and provider.
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
        # Fall back to scratchpad mode when the selected model explicitly does NOT
        # support function calling (e.g. Qwen 2.5 Coder on NVIDIA) to avoid malformed
        # tool-call responses leaking into the final answer.
        from agent_eval.config import get_model_profile

        profile = get_model_profile(self.config.model) if self.config.model else None
        if profile and not profile.supports_function_calling:
            self._use_function_calling = False
        else:
            self._use_function_calling = len(self.tools.list_tools()) > 0

    def run(self, task: str) -> tuple[str, RunRecord]:
        run = self._make_run(task)
        self.recorder.start_run(run)

        messages: list[Message] = self._build_initial_messages(task)

        final_answer: str | None = None
        tool_call_count = 0
        try:
            for step in range(1, self.config.max_steps + 1):
                run.total_steps = step
                thought_text, message, is_final = self._reason_step(step, messages, run)

                if is_final or message.role == Role.ASSISTANT and not message.tool_calls:
                    content = message.content or ""
                    if content.strip():
                        final_answer = self._clean_final_answer(content)
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
                    tool_call_count += 1
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

                # If we've used 3+ tools, force the next step to be a final answer
                if tool_call_count >= 3 and self._use_function_calling:
                    hint = system_message(
                        "You have used sufficient tools. Now synthesize all the information gathered "
                        "and provide your final answer. Do NOT call any more tools."
                    )
                    messages.append(hint)

            else:
                # Exhausted max_steps without producing a final answer
                if not final_answer:
                    truncated = self._extract_last_content(messages) or ""
                    final_answer = f"[Agent exceeded max steps ({self.config.max_steps})]"
                    run = self._finalize_run(
                        run,
                        output=truncated or final_answer,
                        status=RunStatus.FAILED,
                        error=f"Agent exceeded max steps ({self.config.max_steps}) without final answer",
                    )
                    return (final_answer, run)
        except (RuntimeError, ValueError, ConnectionError, TimeoutError, OSError, KeyError, AttributeError) as e:
            logger.exception(f"Agent run failed: {e}")
            cleaned = self._clean_final_answer(final_answer) if final_answer else None
            self._finalize_run(
                run, output=cleaned, status=RunStatus.FAILED, error=f"{type(e).__name__}: {e}"
            )
            return (cleaned or f"Error: {e}", run)

        cleaned = self._clean_final_answer(final_answer) if final_answer else ""
        run = self._finalize_run(run, output=cleaned, status=RunStatus.SUCCESS)
        return (cleaned or "", run)

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
        except (RuntimeError, ValueError, ConnectionError, TimeoutError, OSError) as e:
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
            # NVIDIA API requires: assistant message must have either content or tool_calls, not both
            return thought, assistant_message(content=None, tool_calls=tool_calls), False

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
            # NVIDIA API requires: assistant message must have either content or tool_calls, not both
            return (
                thought,
                assistant_message(
                    content=None,
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
                return ReActAgent._clean_final_answer(m.content)
        return None

    @staticmethod
    def _clean_final_answer(text: str) -> str:
        """Strip internal Thought/Action reasoning from a final answer."""
        if not text:
            return text
        # Remove Thought sections
        text = re.sub(r"Thought:\s*.+?(?=\nAction:|\Z)", "", text, flags=re.DOTALL)
        # Remove Action sections (but keep Action Input / Final Answer)
        text = re.sub(r"Action:\s*.+?(?=\nAction Input:|\nFinal Answer:|\Z)", "", text, flags=re.DOTALL)
        # If there's a "Final Answer:" section, extract only that
        final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if final_match:
            text = final_match.group(1).strip()
        elif "Action Input:" in text:
            # Try to extract Action Input content
            input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
            if input_match:
                text = input_match.group(1).strip()
        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        # If after cleaning the text is empty, return original
        if not text or len(text) < 2:
            return text.strip()
        return text
