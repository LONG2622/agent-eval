"""Configuration loader with environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env with explicit path resolution so it works regardless of CWD
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path))
else:
    load_dotenv()  # fallback: search upward from CWD


# -------------------- Pydantic Config Schemas --------------------


class LLMRetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_factor: float = 2.0


class LLMModelProfile(BaseModel):
    """A selectable model with its own api_key/base_url/provider."""
    id: str = Field(..., description="Unique id used in API/UI (e.g. 'tju-llm')")
    display_name: str = Field(..., description="Human-friendly name shown in UI")
    provider: str = "openai"
    model: str = Field(..., description="Underlying model name sent to the API")
    api_key: str = ""
    base_url: str = ""
    description: str = ""
    supports_function_calling: bool = True
    supports_chinese: bool = True


class LLMConfig(BaseModel):
    default_provider: str = "openai"
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    retry: LLMRetryConfig = Field(default_factory=LLMRetryConfig)
    model_profiles: list[LLMModelProfile] = Field(
        default_factory=list, description="All selectable models"
    )


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


def load_config(config_path: str | Path | None = None, force_reload: bool = False) -> AppConfig:
    """Load configuration from YAML file with env var substitution."""
    global _config_instance
    if _config_instance is not None and not force_reload:
        return _config_instance

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    substituted = _walk_and_substitute(raw)
    _config_instance = AppConfig(**substituted)

    # Populate model_profiles from env vars (so .env is the single source of truth)
    _config_instance.llm.model_profiles = _build_model_profiles_from_env()

    # Override default_model in case default.yaml used a different placeholder
    env_default = os.environ.get("LLM_DEFAULT_MODEL") or os.environ.get("OPENAI_MODEL")
    if env_default:
        _config_instance.llm.default_model = env_default
    # Also: if OPENAI_* env vars are set but no profile matches default_model,
    # create a fallback profile so the default model is always usable
    if not _find_profile_for_model(_config_instance.llm.model_profiles, _config_instance.llm.default_model):
        fallback_key = os.environ.get("OPENAI_API_KEY", "")
        fallback_url = os.environ.get("OPENAI_BASE_URL", "")
        if fallback_key or fallback_url:
            _config_instance.llm.model_profiles.insert(
                0,
                LLMModelProfile(
                    id=_config_instance.llm.default_model,
                    display_name=_config_instance.llm.default_model,
                    model=_config_instance.llm.default_model,
                    api_key=fallback_key,
                    base_url=fallback_url,
                    description="Default model from OPENAI_* env vars",
                ),
            )

    # Ensure output directories exist
    for dir_attr in ("output_dir", "trace_dir", "run_dir", "eval_dir"):
        Path(getattr(_config_instance.storage, dir_attr)).mkdir(parents=True, exist_ok=True)

    return _config_instance


def _build_model_profiles_from_env() -> list[LLMModelProfile]:
    """Build selectable model profiles from explicit env vars."""
    profiles: list[LLMModelProfile] = []

    # 1) Tianjin University
    tju_key = os.environ.get("TJU_API_KEY", "")
    tju_url = os.environ.get("TJU_BASE_URL", "")
    tju_model = os.environ.get("TJU_MODEL_NAME", "tju-llm")
    if tju_key or tju_url:
        profiles.append(
            LLMModelProfile(
                id=tju_model,
                display_name="天津大学 (tju-llm)",
                model=tju_model,
                api_key=tju_key,
                base_url=tju_url,
                description="中文友好 + 原生支持 Function Calling，推荐日常使用",
                supports_function_calling=True,
                supports_chinese=True,
            )
        )

    # 2) NVIDIA A: Llama 3.1 70B
    nvk1_key = os.environ.get("NVIDIA_LLAMA_API_KEY", "")
    nvk1_url = os.environ.get("NVIDIA_LLAMA_BASE_URL", "")
    nvk1_model = os.environ.get("NVIDIA_LLAMA_MODEL_NAME", "")
    if (nvk1_key or nvk1_url) and nvk1_model:
        profiles.append(
            LLMModelProfile(
                id=nvk1_model,
                display_name="NVIDIA: Meta Llama 3.1 70B Instruct",
                model=nvk1_model,
                api_key=nvk1_key,
                base_url=nvk1_url,
                description="英文推理能力强，中文能力有限",
                supports_function_calling=True,
                supports_chinese=False,
            )
        )

    # 3) NVIDIA B: Mistral Nemotron (Chinese-friendly, no FC)
    nvk2_key = os.environ.get("NVIDIA_QWEN_API_KEY", "")
    nvk2_url = os.environ.get("NVIDIA_QWEN_BASE_URL", "")
    nvk2_model = os.environ.get("NVIDIA_QWEN_MODEL_NAME", "")
    if (nvk2_key or nvk2_url) and nvk2_model:
        profiles.append(
            LLMModelProfile(
                id=nvk2_model,
                display_name="NVIDIA: Mistral Nemotron",
                model=nvk2_model,
                api_key=nvk2_key,
                base_url=nvk2_url,
                description="中文友好 + 推理能力强，不支持 Function Calling，Agent 自动降级到 Scratchpad 模式",
                supports_function_calling=False,
                supports_chinese=True,
            )
        )

    return profiles


def _find_profile_for_model(
    profiles: list[LLMModelProfile], model: str
) -> LLMModelProfile | None:
    """Lookup a profile by id first, then by underlying model name."""
    for p in profiles:
        if p.id == model:
            return p
    for p in profiles:
        if p.model == model:
            return p
    return None


def get_model_profile(model: str | None = None) -> LLMModelProfile | None:
    """Public helper: resolve a model selector string to its profile."""
    cfg = load_config()
    model = model or cfg.llm.default_model
    return _find_profile_for_model(cfg.llm.model_profiles, model)


def list_model_profiles() -> list[LLMModelProfile]:
    """Return all registered selectable model profiles."""
    cfg = load_config()
    return list(cfg.llm.model_profiles)


def reset_config() -> None:
    """Clear the cached config so the next load_config() reads fresh data."""
    global _config_instance
    _config_instance = None


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
