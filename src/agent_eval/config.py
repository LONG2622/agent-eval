"""Configuration loader with environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# -------------------- Pydantic Config Schemas --------------------


class LLMRetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_factor: float = 2.0


class LLMConfig(BaseModel):
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)


class AgentConfig(BaseModel):
    default_type: str = "react"
    max_steps: int = 10
    step_timeout_seconds: int = 60


class PricingEntry(BaseModel):
    prompt: float = 0.0
    completion: float = 0.0


class StorageConfig(BaseModel):
    backend: str = "jsonl"
    output_dir: str = "./outputs"
    trace_dir: str = "./outputs/traces"
    run_dir: str = "./outputs/runs"
    eval_dir: str = "./outputs/evaluations"


class EvaluationQualityConfig(BaseModel):
    keyword_match_threshold: float = 0.6


class JudgeConfig(BaseModel):
    """Configuration for LLM-as-Judge evaluator."""
    enabled: bool = False
    model: str = "tju-llm"
    temperature: float = 0.1
    max_tokens: int = 1500


class EvaluationConfig(BaseModel):
    default_evaluators: list[str] = Field(
        default_factory=lambda: [
            "success_rate",
            "tool_usage",
            "latency",
            "token_cost",
            "answer_quality_keyword",
        ]
    )
    quality: EvaluationQualityConfig = Field(default_factory=EvaluationQualityConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class AppConfig(BaseModel):
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pricing: dict[str, PricingEntry] = Field(default_factory=dict)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# -------------------- Env Var Substitution --------------------

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR} or ${VAR:-default} patterns with environment values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(var_name, default)

    return _ENV_PATTERN.sub(_replace, value)


def _walk_and_substitute(obj: Any) -> Any:
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


# -------------------- Loader --------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"

_config_instance: AppConfig | None = None


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file with env var substitution."""
    global _config_instance
    if _config_instance is not None:
        return _config_instance

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    substituted = _walk_and_substitute(raw)
    _config_instance = AppConfig(**substituted)

    # Ensure output directories exist
    for dir_attr in ("output_dir", "trace_dir", "run_dir", "eval_dir"):
        Path(getattr(_config_instance.storage, dir_attr)).mkdir(parents=True, exist_ok=True)

    return _config_instance


def get_pricing(model: str) -> PricingEntry:
    """Get pricing entry for a given model, trying several name variants."""
    cfg = load_config()
    if model in cfg.pricing:
        return cfg.pricing[model]
    # Try fuzzy match: gpt-4o-mini-2024-07-18 -> gpt-4o-mini
    for key, entry in cfg.pricing.items():
        if model.startswith(key):
            return entry
    return PricingEntry(prompt=0.0, completion=0.0)
