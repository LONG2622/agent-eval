"""Tests for config loading and environment variable substitution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_eval.config import (
    LLMModelProfile,
    _find_profile_for_model,
    _substitute_env_vars,
    _walk_and_substitute,
    get_model_profile,
    get_pricing,
    list_model_profiles,
    load_config,
    reset_config,
)


class TestEnvVarSubstitution:
    def test_substitute_simple(self, monkeypatch):
        """Should substitute ${VAR} with env value."""
        monkeypatch.setenv("TEST_VAR", "hello")
        result = _substitute_env_vars("prefix_${TEST_VAR}_suffix")
        assert result == "prefix_hello_suffix"

    def test_substitute_default(self, monkeypatch):
        """Should use default when env var is missing."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = _substitute_env_vars("${MISSING_VAR:-default_val}")
        assert result == "default_val"

    def test_substitute_missing_no_default(self, monkeypatch):
        """Missing var without default should produce empty string."""
        monkeypatch.delenv("NO_VAR", raising=False)
        result = _substitute_env_vars("${NO_VAR}")
        assert result == ""

    def test_walk_nested(self, monkeypatch):
        """Should substitute vars in nested dicts and lists."""
        monkeypatch.setenv("KEY", "value")
        data = {
            "outer": "${KEY}",
            "nested": {"inner": "${KEY}"},
            "list": ["${KEY}", "static"],
        }
        result = _walk_and_substitute(data)
        assert result["outer"] == "value"
        assert result["nested"]["inner"] == "value"
        assert result["list"][0] == "value"
        assert result["list"][1] == "static"


class TestLoadConfig:
    def test_load_config_returns_app_config(self):
        """load_config should return AppConfig."""
        reset_config()
        cfg = load_config(force_reload=True)
        assert cfg is not None
        assert cfg.agent.default_type == "react"
        assert cfg.agent.max_steps == 10

    def test_load_config_cached(self):
        """Second call should return cached instance."""
        reset_config()
        cfg1 = load_config()
        cfg2 = load_config()
        assert cfg1 is cfg2

    def test_force_reload(self):
        """force_reload should create new instance."""
        reset_config()
        cfg1 = load_config()
        cfg2 = load_config(force_reload=True)
        assert cfg1 is not cfg2

    def test_config_defaults(self):
        """Default config should have sensible defaults."""
        cfg = load_config(force_reload=True)
        assert cfg.llm.temperature == 0.8
        assert cfg.llm.max_tokens == 20000
        assert cfg.storage.backend == "jsonl"
        assert cfg.evaluation.default_evaluators  # non-empty list

    def test_evaluation_quality_config(self):
        """Keyword match threshold should be configurable."""
        cfg = load_config(force_reload=True)
        assert 0 <= cfg.evaluation.quality.keyword_match_threshold <= 1.0


class TestModelProfiles:
    def test_list_model_profiles(self):
        """list_model_profiles should return list."""
        profiles = list_model_profiles()
        assert isinstance(profiles, list)

    def test_find_profile_by_id(self):
        """Should find profile by id."""
        profiles = [
            LLMModelProfile(
                id="test-model",
                display_name="Test",
                model="test-model",
                api_key="key",
                base_url="http://test.com",
            )
        ]
        result = _find_profile_for_model(profiles, "test-model")
        assert result is not None
        assert result.id == "test-model"

    def test_find_profile_by_model(self):
        """Should find profile by underlying model name."""
        profiles = [
            LLMModelProfile(
                id="custom-id",
                display_name="Test",
                model="actual-model-name",
                api_key="key",
                base_url="http://test.com",
            )
        ]
        result = _find_profile_for_model(profiles, "actual-model-name")
        assert result is not None

    def test_find_profile_not_found(self):
        """Should return None when not found."""
        profiles = []
        result = _find_profile_for_model(profiles, "nonexistent")
        assert result is None


class TestGetPricing:
    def test_get_pricing_default(self):
        """get_pricing should return default for unknown model."""
        reset_config()
        pricing = get_pricing("unknown_model")
        assert pricing.prompt == 0.0
        assert pricing.completion == 0.0

    def test_get_pricing_known(self, monkeypatch):
        """get_pricing should return known pricing."""
        reset_config()
        monkeypatch.setenv("PRICING_GPT_4O_MINI_PROMPT", "0.00015")
        monkeypatch.setenv("PRICING_GPT_4O_MINI_COMPLETION", "0.0006")
        cfg = load_config(force_reload=True)
        pricing = get_pricing("gpt-4o-mini")
        assert pricing.prompt == 0.00015
        assert pricing.completion == 0.0006


class TestResetConfig:
    def test_reset(self):
        """reset_config should clear cache."""
        reset_config()
        cfg1 = load_config()
        reset_config()
        cfg2 = load_config()
        assert cfg1 is not cfg2