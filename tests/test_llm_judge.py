"""Comprehensive tests for the LLM-as-Judge evaluator.

Covers:
  - LLMJudgeEvaluator constructor (with/without gateway, with/without model)
  - JudgeOutput.to_evaluator_results shape and score normalization
  - _parse_judge_response: direct JSON, markdown fences, JSON-in-prose, complete-failure
  - evaluate(): normal path, empty_output short-circuit, parse-error fallback, RuntimeError propagation
  - _call_judge() kwargs (model, temperature, max_tokens) passed through to gateway
  - Integration: get_builtin_evaluator_instances with "llm_judge"
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_eval.evaluation.base import EvalDimension, SubMetric
from agent_eval.evaluation.builtin import get_builtin_evaluator_instances
from agent_eval.evaluation.llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeDimensionScore,
    JudgeOutput,
    LLMJudgeEvaluator,
)
from agent_eval.llm.gateway import LLMGateway
from agent_eval.llm.messages import Message
from agent_eval.llm.providers.base import (
    LLMCallOptions,
    LLMProvider,
    LLMResponse,
)
from agent_eval.trace.models import RunRecord, RunStatus, TokenUsage

# ============================================================
# Fake infrastructure
# ============================================================


class RecordingProvider(LLMProvider):
    """Scripted LLMProvider that returns a canned response and records call args."""

    name = "fake"

    def __init__(self, response_content: str | None = None, *, exc: Exception | None = None):
        self._response_content = response_content
        self._exc = exc
        self.call_count = 0
        self.last_messages: list[Message] | None = None
        self.last_options: LLMCallOptions | None = None

    def chat(self, messages, options=None) -> LLMResponse:
        self.call_count += 1
        self.last_messages = list(messages)
        self.last_options = options
        if self._exc is not None:
            raise self._exc
        return LLMResponse(
            content=self._response_content,
            model="judge-test-model",
            prompt_tokens=120,
            completion_tokens=80,
            total_tokens=200,
        )


def _build_judge_json(
    correctness: int = 4,
    relevance: int = 4,
    completeness: int = 3,
    harmlessness: int = 5,
    readability: int = 4,
) -> str:
    """Construct a valid JudgeOutput JSON string for the FakeProvider."""
    data = {
        "correctness":  {"dimension": "correctness",  "score": correctness,  "reason": "looks correct"},
        "relevance":    {"dimension": "relevance",    "score": relevance,    "reason": "on-topic"},
        "completeness": {"dimension": "completeness", "score": completeness, "reason": "covers key points"},
        "harmlessness": {"dimension": "harmlessness", "score": harmlessness, "reason": "no issues"},
        "readability":  {"dimension": "readability",  "score": readability,  "reason": "easy to follow"},
    }
    return json.dumps(data, ensure_ascii=False)


def _make_run(final_output: str | None = "37", expected_output: str | None = "37") -> RunRecord:
    """Convenience helper: build a RunRecord for LLM-judge testing."""
    return RunRecord(
        run_id="judge_test_run_001",
        task_id="task_001",
        agent_name="test_agent",
        input_text="Calculate sqrt(144) + 5^2",
        status=RunStatus.SUCCESS,
        final_output=final_output,
        total_steps=2,
        total_latency_ms=1234,
        tokens=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        total_cost=0.0001,
        expected_output=expected_output,
    )


# ============================================================
# Helper evaluator factory
# ============================================================


def _make_evaluator(provider: RecordingProvider, judge_model: str | None = None) -> LLMJudgeEvaluator:
    gateway = LLMGateway(provider=provider)
    return LLMJudgeEvaluator(judge_gateway=gateway, judge_model=judge_model)


# ============================================================
# 1. Constructor tests
# ============================================================


class TestLLMJudgeEvaluatorConstructor:
    def test_uses_provided_gateway_directly(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        gateway = LLMGateway(provider=provider)
        ev = LLMJudgeEvaluator(judge_gateway=gateway, judge_model="my-judge")
        assert ev._gateway is gateway
        assert ev._judge_model == "my-judge"

    def test_default_gateway_created_when_none_passed(self, patch_config, monkeypatch):
        # Patch LLMGateway to record constructor args
        original_init = LLMGateway.__init__

        def spy_init(self_, *args, **kwargs):
            original_init(self_, *args, **kwargs)
            self_._created = True

        monkeypatch.setattr(LLMGateway, "__init__", spy_init)
        ev = LLMJudgeEvaluator()
        assert ev._gateway is not None
        assert getattr(ev._gateway, "_created", False) is True

    def test_judge_model_defaults_to_config_llm_default_model(self, patch_config):
        cfg = patch_config
        provider = RecordingProvider(response_content=_build_judge_json())
        gateway = LLMGateway(provider=provider)
        ev = LLMJudgeEvaluator(judge_gateway=gateway)
        # When judge_model is not passed, falls back to cfg.llm.default_model
        assert ev._judge_model == cfg.llm.default_model

    def test_class_attributes(self, patch_config):
        ev = LLMJudgeEvaluator(judge_gateway=LLMGateway(provider=RecordingProvider("{}")))
        assert ev.name == "llm_judge"
        assert ev.dimension == EvalDimension.ANSWER_QUALITY


# ============================================================
# 2. JudgeOutput.to_evaluator_results
# ============================================================


class TestJudgeOutputToEvaluatorResults:
    def test_returns_six_results(self):
        jo = JudgeOutput.model_validate_json(_build_judge_json(4, 4, 3, 5, 4))
        results = jo.to_evaluator_results("rid_1", "llm_judge")
        assert len(results) == 6  # 5 facets + 1 overall

    def test_score_normalization_formula(self):
        # score 1 -> (1-1)/4 = 0.0, score 5 -> (5-1)/4 = 1.0
        jo = JudgeOutput(
            correctness=JudgeDimensionScore(dimension="correctness", score=5, reason="best"),
            relevance=JudgeDimensionScore(dimension="relevance", score=1, reason="worst"),
            completeness=JudgeDimensionScore(dimension="completeness", score=3, reason="mid"),
            harmlessness=JudgeDimensionScore(dimension="harmlessness", score=5, reason="safe"),
            readability=JudgeDimensionScore(dimension="readability", score=2, reason="poor"),
        )
        results = {
            (r.sub_metric.value if r.sub_metric else None): r
            for r in jo.to_evaluator_results("rid", "llm_judge")
        }
        # correctness: 5 -> 1.0
        assert results["correctness"].score == pytest.approx(1.0, abs=0.001)
        assert results["correctness"].passed is True
        # relevance: 1 -> 0.0
        assert results["relevance"].score == pytest.approx(0.0, abs=0.001)
        assert results["relevance"].passed is False
        # completeness: 3 -> 0.5
        assert results["completeness"].score == pytest.approx(0.5, abs=0.001)
        assert results["completeness"].passed is False

    def test_passed_threshold_is_0_6(self):
        # normalized >= 0.6 means raw score >= 1 + 0.6*4 = 3.4 -> raw >= 4
        jo = JudgeOutput(
            correctness=JudgeDimensionScore(dimension="correctness", score=4, reason="ok"),  # 0.75 >= 0.6
            relevance=JudgeDimensionScore(dimension="relevance", score=3, reason="mid"),   # 0.5  < 0.6
            completeness=JudgeDimensionScore(dimension="completeness", score=4, reason="ok"),
            harmlessness=JudgeDimensionScore(dimension="harmlessness", score=5, reason="safe"),
            readability=JudgeDimensionScore(dimension="readability", score=4, reason="ok"),
        )
        results = jo.to_evaluator_results("rid", "llm_judge")
        by_sub = {
            (r.sub_metric.value if r.sub_metric else None): r
            for r in results
        }
        assert by_sub["correctness"].passed is True
        assert by_sub["relevance"].passed is False
        # overall: (4+3+4+5+4)/5 = 4.0 -> (4-1)/4 = 0.75 >= 0.6
        # Hmm, overall result also has sub_metric=None... let's find it by aggregations
        overall_res = [r for r in results if r.details and r.details.get("aggregation") == "mean_of_5_dims"][0]
        assert overall_res.score == pytest.approx(0.75, abs=0.001)
        assert overall_res.passed is True

    def test_sub_metric_mapping(self):
        jo = JudgeOutput.model_validate_json(_build_judge_json())
        results = jo.to_evaluator_results("rid", "llm_judge")
        sub_metric_vals = {
            (r.sub_metric.value if r.sub_metric else None): r
            for r in results
        }
        assert sub_metric_vals["correctness"].sub_metric == SubMetric.CORRECTNESS
        assert sub_metric_vals["relevance"].sub_metric == SubMetric.RELEVANCE
        assert sub_metric_vals["completeness"].sub_metric == SubMetric.COMPLETENESS
        # harmlessness/readability: no built-in sub-metric -> None
        assert results[3].details["dimension"] == "harmlessness"
        assert results[4].details["dimension"] == "readability"
        assert results[3].sub_metric is None
        assert results[4].sub_metric is None

    def test_result_metadata_shape(self):
        jo = JudgeOutput.model_validate_json(_build_judge_json(correctness=5, relevance=4, completeness=3, harmlessness=5, readability=4))
        results = jo.to_evaluator_results("run_abc", "llm_judge")
        for r in results:
            assert r.run_id == "run_abc"
            assert r.evaluator == "llm_judge"
            assert r.dimension == EvalDimension.ANSWER_QUALITY
            assert isinstance(r.details, dict)
        # Overall result
        overall = [r for r in results if r.details and r.details.get("aggregation") == "mean_of_5_dims"][0]
        assert overall.details["judge_model"] == "llm"
        assert overall.details["sub_scores"]["correctness"] == 5
        assert overall.details["sub_scores"]["harmlessness"] == 5


# ============================================================
# 3. _parse_judge_response - all parse paths
# ============================================================


class TestParseJudgeResponse:
    """Cover every branch of LLMJudgeEvaluator._parse_judge_response."""

    def test_direct_clean_json(self):
        text = _build_judge_json(correctness=4, relevance=4, completeness=4, harmlessness=5, readability=4)
        jo = LLMJudgeEvaluator._parse_judge_response(text)
        assert isinstance(jo, JudgeOutput)
        assert jo.correctness.score == 4

    def test_json_in_markdown_fence_with_lang_tag(self):
        inner = _build_judge_json(correctness=3)
        text = f"```json\n{inner}\n```"
        jo = LLMJudgeEvaluator._parse_judge_response(text)
        assert jo.correctness.score == 3

    def test_json_in_markdown_fence_without_lang(self):
        inner = _build_judge_json(readability=5)
        text = f"```\n{inner}\n```"
        jo = LLMJudgeEvaluator._parse_judge_response(text)
        assert jo.readability.score == 5

    def test_json_surrounded_by_prose_step2_regex_extract(self):
        inner = _build_judge_json(relevance=2)
        text = (
            "Here is my assessment:\n"
            f"{inner}\n"
            "Hope this helps you judge the quality."
        )
        jo = LLMJudgeEvaluator._parse_judge_response(text)
        assert jo.relevance.score == 2

    def test_complete_failure_raises(self):
        """Totally non-JSON text at all stages -> json.loads raises JSONDecodeError."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            LLMJudgeEvaluator._parse_judge_response("I don't have any scores for you.")

    def test_partial_json_missing_fields_still_fails(self):
        """If one dimension key is entirely missing, Pydantic validation fails."""
        incomplete = json.dumps({
            "correctness":  {"dimension": "correctness",  "score": 5, "reason": "yes"},
            "relevance":    {"dimension": "relevance",    "score": 5, "reason": "yes"},
            # completeness, harmlessness, readability missing
        })
        with pytest.raises((ValidationError, ValueError)):
            LLMJudgeEvaluator._parse_judge_response(incomplete)

    def test_score_out_of_range_fails(self):
        """JudgeDimensionScore.score is constrained to 1..5 via Field(ge=1, le=5)."""
        bad = _build_judge_json(correctness=7)  # out of range
        with pytest.raises((ValidationError, ValueError)):
            LLMJudgeEvaluator._parse_judge_response(bad)


# ============================================================
# 4. evaluate() - happy path
# ============================================================


class TestEvaluateHappyPath:
    def test_returns_list_of_results(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json(4, 4, 4, 5, 4))
        ev = _make_evaluator(provider)
        run = _make_run()
        results = ev.evaluate(run, [])
        assert isinstance(results, list)
        assert len(results) == 6  # 5 facets + 1 overall

    def test_evaluate_passes_expected_output_as_reference(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        ev = _make_evaluator(provider)
        run = _make_run(final_output="37", expected_output="37")
        ev.evaluate(run, [])
        assert provider.call_count == 1
        # The last message should be the user prompt containing expected="37"
        user_msg = provider.last_messages[-1]
        assert isinstance(user_msg.content, str)
        assert "### Expected Answer" in user_msg.content
        assert "37" in user_msg.content

    def test_expected_output_missing_defaults_to_not_provided(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        ev = _make_evaluator(provider)
        run = _make_run(expected_output=None)
        ev.evaluate(run, [])
        user_msg = provider.last_messages[-1]
        assert "(not provided)" in user_msg.content

    def test_calls_gateway_with_correct_kwargs(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        ev = _make_evaluator(provider, judge_model="tju-llm-v2")
        run = _make_run()
        ev.evaluate(run, [])
        assert provider.last_options is not None
        assert provider.last_options.model == "tju-llm-v2"
        assert provider.last_options.temperature == pytest.approx(0.1)
        assert provider.last_options.max_tokens == 1500


# ============================================================
# 5. evaluate() - empty output short-circuit
# ============================================================


class TestEvaluateEmptyOutput:
    def test_none_final_output_returns_zero_results(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        ev = _make_evaluator(provider)
        run = _make_run(final_output=None)
        results = ev.evaluate(run, [])
        # Should short-circuit BEFORE calling gateway
        assert provider.call_count == 0
        assert len(results) == 2
        for r in results:
            assert r.score == 0.0
            assert r.passed is False
        # COMPLETENESS sub_metric result
        completeness_result = [r for r in results if r.sub_metric == SubMetric.COMPLETENESS]
        assert len(completeness_result) == 1
        assert completeness_result[0].details["reason"] == "empty_final_output"

    def test_whitespace_only_final_output_treated_as_empty(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        ev = _make_evaluator(provider)
        run = _make_run(final_output="   \n  ")
        results = ev.evaluate(run, [])
        assert provider.call_count == 0
        assert len(results) == 2


# ============================================================
# 6. evaluate() - parse error triggers fallback
# ============================================================


class TestEvaluateFallback:
    def test_invalid_json_uses_fallback(self, patch_config):
        provider = RecordingProvider(response_content="this is not json at all")
        ev = _make_evaluator(provider)
        run = _make_run()
        results = ev.evaluate(run, [])
        # Should NOT have raised — fallback kicks in
        assert isinstance(results, list)
        assert len(results) >= 1
        # All result evaluator names are rewritten to "llm_judge"
        assert all(r.evaluator == "llm_judge" for r in results)
        # Each result should carry metadata["fallback"] = True
        assert all(r.metadata.get("fallback") is True for r in results)

    def test_malformed_json_object_uses_fallback(self, patch_config):
        """Partial JSON like {"correctness": 9} (missing nested reasons) — still fails."""
        malformed = '{"correctness": 9, "relevance": 8}'
        provider = RecordingProvider(response_content=malformed)
        ev = _make_evaluator(provider)
        run = _make_run()
        results = ev.evaluate(run, [])
        assert isinstance(results, list)
        assert all(r.evaluator == "llm_judge" for r in results)
        assert all(r.metadata.get("fallback") is True for r in results)


# ============================================================
# 7. evaluate() - RuntimeError propagates (NOT caught by evaluate)
# ============================================================


class TestEvaluateProviderError:
    def test_runtime_error_from_provider_propagates(self, patch_config):
        """LLMJudgeEvaluator.evaluate only catches (JSONDecodeError, ValueError, TypeError, KeyError).
        RuntimeError from a broken provider propagates out of evaluate()."""
        provider = RecordingProvider(exc=RuntimeError("network down"))
        ev = _make_evaluator(provider)
        run = _make_run()
        with pytest.raises(RuntimeError, match="network down"):
            ev.evaluate(run, [])


# ============================================================
# 8. Integration: engine and default evaluator discovery
# ============================================================


class TestIntegrationGetBuiltinEvaluators:
    def test_llm_judge_included_when_name_requested(self, patch_config):
        instances = get_builtin_evaluator_instances([
            "success_rate", "llm_judge"
        ])
        names = [e.name for e in instances]
        assert "success_rate" in names
        assert "llm_judge" in names
        llm_judges = [e for e in instances if e.name == "llm_judge"]
        assert len(llm_judges) == 1
        assert isinstance(llm_judges[0], LLMJudgeEvaluator)

    def test_llm_judge_not_in_default_evaluators_list(self, patch_config):
        """Config default_evaluators does NOT include 'llm_judge'.
        It must be explicitly requested."""
        from agent_eval.config import load_config
        cfg = load_config()
        assert "llm_judge" not in cfg.evaluation.default_evaluators


# ============================================================
# 9. _call_judge() prompt assembly
# ============================================================


class TestCallJudgePromptAssembly:
    def test_system_prompt_first_user_second(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        ev = _make_evaluator(provider, judge_model="tju-llm")
        run = _make_run(final_output="42", expected_output="42")
        ev.evaluate(run, [])
        assert provider.last_messages is not None
        system_msg = provider.last_messages[0]
        user_msg = provider.last_messages[1]
        assert system_msg.role.value == "system"
        assert system_msg.content == JUDGE_SYSTEM_PROMPT
        assert user_msg.role.value == "user"
        assert "42" in user_msg.content
        assert "Calculate sqrt(144) + 5^2" in user_msg.content


# ============================================================
# 10. _empty_output_result shape
# ============================================================


class TestEmptyOutputResult:
    def test_shape_matches_contract(self, patch_config):
        provider = RecordingProvider(response_content=_build_judge_json())
        ev = _make_evaluator(provider)
        results = ev._empty_output_result("rid_empty")
        assert len(results) == 2
        completeness = [r for r in results if r.sub_metric == SubMetric.COMPLETENESS]
        overall = [r for r in results if r.sub_metric is None]
        assert len(completeness) == 1
        assert len(overall) == 1
        for r in results:
            assert r.run_id == "rid_empty"
            assert r.evaluator == "llm_judge"
            assert r.dimension == EvalDimension.ANSWER_QUALITY
            assert r.score == 0.0
            assert r.passed is False
            assert r.details["reason"] == "empty_final_output"


# ============================================================
# 11. JudgeDimensionScore model validation
# ============================================================


class TestJudgeDimensionScoreModel:
    def test_default_reason_empty_string(self):
        s = JudgeDimensionScore(dimension="correctness", score=4)
        assert s.reason == ""
        assert s.dimension == "correctness"
        assert s.score == 4

    def test_score_boundary_ok(self):
        JudgeDimensionScore(dimension="x", score=1)
        JudgeDimensionScore(dimension="x", score=5)

    def test_score_out_of_boundary_rejected(self):
        with pytest.raises(ValidationError):
            JudgeDimensionScore(dimension="x", score=0)
        with pytest.raises(ValidationError):
            JudgeDimensionScore(dimension="x", score=6)
