"""Token counting and cost calculation utilities."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from agent_eval.config import get_pricing

logger = logging.getLogger("agent_eval.llm.tokenizer")


# -------------------- Tokenizer --------------------


@lru_cache(maxsize=32)
def _get_encoder(model: str):
    """Get tiktoken encoder for a model. Falls back to cl100k_base on failure."""
    try:
        import tiktoken

        return tiktoken.encoding_for_model(model)
    except (ImportError, ModuleNotFoundError):
        try:
            import tiktoken

            return tiktoken.get_encoding("cl100k_base")
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning(f"Failed to load tiktoken, using char-based estimator: {e}")
            return None


def count_text_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in a string using tiktoken if available."""
    if not text:
        return 0
    encoder = _get_encoder(model)
    if encoder is not None:
        return len(encoder.encode(text))
    # Fallback: estimate ~4 chars per token
    return max(1, len(text) // 4)


def count_message_tokens(messages: list[dict[str, Any]], model: str = "gpt-4o-mini") -> int:
    """Count tokens for a list of messages (OpenAI format).

    Follows the formula used by OpenAI docs:
      every message adds 4 tokens (role, content, etc.) + content
    """
    encoder = _get_encoder(model)

    def _encode(text: str) -> int:
        if not text:
            return 0
        if encoder is not None:
            return len(encoder.encode(text))
        return max(1, len(text) // 4)

    total = 0
    for msg in messages:
        total += 4  # base per-message overhead
        for key, value in msg.items():
            if isinstance(value, str):
                total += _encode(value)
            elif key == "tool_calls" and isinstance(value, list):
                for tc in value:
                    fn = tc.get("function", {})
                    total += _encode(fn.get("name", ""))
                    total += _encode(str(fn.get("arguments", {})))
    total += 2  # final assistant priming
    return total


def count_tokens_breakdown(
    messages: list[dict[str, Any]],
    completion_text: str,
    model: str = "gpt-4o-mini",
) -> dict[str, int]:
    """Return {prompt_tokens, completion_tokens, total_tokens}."""
    prompt_tokens = count_message_tokens(messages, model)
    completion_tokens = count_text_tokens(completion_text or "", model)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# -------------------- Cost Calculation --------------------


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float:
    """Calculate USD cost for a request based on pricing table."""
    pricing = get_pricing(model)
    return round(
        (prompt_tokens / 1000) * pricing.prompt
        + (completion_tokens / 1000) * pricing.completion,
        6,
    )
