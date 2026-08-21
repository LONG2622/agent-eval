"""Built-in example tools shipped with the framework."""

from __future__ import annotations

import datetime
import math
import random
import re
from typing import Any

from agent_eval.logger import setup_logger
from agent_eval.tools.registry import ToolResult, tool

logger = setup_logger("agent_eval.tools.builtin")


# -------------------- Calculator --------------------


@tool(name="calculator")
def calculator(expression: str) -> Any:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A math expression string, e.g. "(3 + 4) * 2" or "sqrt(16)".
                    Supports +-*/, parentheses, and basic functions:
                    sqrt, sin, cos, tan, log, exp, pi, e, abs, pow.
    """
    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "pow": pow,
    }
    # Normalize user-friendly syntax to Python
    expr = expression.replace("^", "**")
    # Allow digits, whitespace, + - * / ( ) . , ^(already converted), letters, %
    if not re.match(r"^[\d\s+\-*/().,%a-zA-Z]*$", expr):
        return ToolResult(
            output="", success=False, error="Expression contains disallowed characters"
        )
    try:
        result = eval(expr, safe_globals, {})  # noqa: S307
        return str(result)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, SyntaxError, NameError) as e:
        return ToolResult(output="", success=False, error=f"Eval failed: {e}")


# -------------------- Mock Search --------------------


_MOCK_KNOWLEDGE_BASE: dict[str, str] = {
    "python": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its use of significant whitespace.",
    "pytorch": "PyTorch is an open-source machine learning framework developed primarily by Meta AI. It provides tensor computation with strong GPU acceleration and deep neural networks built on a tape-based autograd system.",
    "llm": "A Large Language Model (LLM) is a type of neural network trained on vast amounts of text data to understand and generate human-like language, based on the Transformer architecture introduced in 2017.",
    "gpt4": "GPT-4 is a large language model developed by OpenAI, released in March 2023. It is a multimodal model capable of processing both text and image inputs, with significantly improved reasoning capabilities over GPT-3.5.",
    "lora": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable low-rank matrices into each layer of the Transformer architecture.",
    "sft": "SFT (Supervised Fine-Tuning) is the process of fine-tuning a pre-trained language model on a dataset of high-quality, human-annotated instruction-response pairs to follow specific instructions.",
    "transformer": "The Transformer architecture, introduced in the 2017 paper 'Attention Is All You Need', is a neural network design based entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "population of beijing": "Beijing has an estimated population of around 21.84 million people as of 2023, making it one of the most populous cities in the world.",
    "capital of france": "The capital of France is Paris, located along the Seine River in the northern part of the country.",
    "year of world war 2": "World War II lasted from 1939 to 1945, beginning with the German invasion of Poland on September 1, 1939, and ending with Japan's formal surrender on September 2, 1945.",
}


@tool(name="web_search")
def web_search(query: str, max_results: int = 3) -> str:
    """Mock web search tool: returns knowledge snippets for known queries.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 3).
    """
    q = query.strip().lower()
    results: list[str] = []

    # Direct key match
    for key, content in _MOCK_KNOWLEDGE_BASE.items():
        if key in q:
            results.append(f"[{key.upper()}] {content}")

    # Fallback: fuzzy keyword scan
    if not results:
        words = re.findall(r"\w+", q)
        scored: list[tuple[int, str, str]] = []
        for key, content in _MOCK_KNOWLEDGE_BASE.items():
            score = sum(1 for w in words if w in (key + " " + content).lower())
            if score > 0:
                scored.append((score, key, content))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, key, content in scored[:max_results]:
            results.append(f"[{key.upper()}] {content}")

    if not results:
        results.append(
            f"No relevant information found for query '{query}'. "
            f"Try asking about: {', '.join(_MOCK_KNOWLEDGE_BASE.keys())}."
        )

    random.shuffle(results)
    return "\n\n".join(results[:max_results])


# -------------------- Get Current Time --------------------


@tool(name="get_current_time")
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: Timezone string (e.g. 'UTC', 'Asia/Shanghai'). Default 'UTC'.
                  Note: In this MVP only UTC and Asia/Shanghai are supported;
                  for other values UTC is returned.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    if timezone.lower() in ("asia/shanghai", "cst", "china", "beijing"):
        now = now + datetime.timedelta(hours=8)
        label = "Asia/Shanghai"
    else:
        label = "UTC"
    return f"Current time ({label}): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"


# -------------------- Read Local File --------------------


@tool(name="read_file")
def read_file(file_path: str, max_chars: int = 5000) -> Any:
    """Read a local text file.

    Args:
        file_path: Path to the file to read.
        max_chars: Maximum number of characters to return (default 5000).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(max_chars)
        suffix = "" if len(content) < max_chars else f"\n... [truncated at {max_chars} chars]"
        return content + suffix
    except FileNotFoundError:
        return ToolResult(output="", success=False, error=f"File not found: {file_path}")
    except UnicodeDecodeError:
        return ToolResult(output="", success=False, error=f"Cannot decode file {file_path} as UTF-8")
    except (OSError, IOError, UnicodeDecodeError) as e:
        return ToolResult(output="", success=False, error=f"Error reading file: {e}")


# -------------------- Registry Populator --------------------


def register_builtin_tools(registry) -> None:
    """Register all built-in tools onto a ToolRegistry instance."""
    for t in (calculator, web_search, get_current_time, read_file):
        registry.register(t)
