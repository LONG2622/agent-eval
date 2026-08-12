"""Five built-in evaluators covering all required dimensions."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

from agent_eval.config import load_config
from agent_eval.evaluation.base import (
    BaseEvaluator,
    EvalDimension,
    EvaluationResult,
    SubMetric,
)
from agent_eval.trace import RunRecord, RunStatus, Span, SpanType

logger = logging.getLogger("agent_eval.evaluation.builtin")


# ============================================================
# 1. Success Rate Evaluator
# ============================================================


class SuccessRateEvaluator(BaseEvaluator):
    """Pass/fail based on run status + keyword match with expected_output."""

    name = "success_rate"
    dimension = EvalDimension.SUCCESS_RATE

    def evaluate(self, run: RunRecord, spans: list[Span]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        # Base success: run finished successfully
        base_passed = run.status == RunStatus.SUCCESS
        final_output = (run.final_output or "").lower()
        expected = (run.expected_output or "").lower()
        keyword_passed = None

        if expected:
            # Keyword match: split expected into non-stop words, check ratio
            expected_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", expected))
            if expected_words:
                output_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", final_output))
                hit = expected_words & output_words
                cfg = load_config()
                threshold = cfg.evaluation.quality.keyword_match_threshold
                ratio = len(hit) / len(expected_words)
                keyword_passed = ratio >= threshold
                results.append(
                    EvaluationResult(
                        run_id=run.run_id,
                        evaluator=self.name,
                        dimension=EvalDimension.ANSWER_QUALITY,
                        sub_metric=SubMetric.KEYWORD_MATCH,
                        score=round(ratio, 4),
                        passed=keyword_passed,
                        details={
                            "expected_keywords": sorted(expected_words),
                            "matched_keywords": sorted(hit),
                            "match_ratio": round(ratio, 4),
                            "threshold": threshold,
                        },
                    )
                )

        passed = base_passed and (keyword_passed if keyword_passed is not None else True)
        results.insert(
            0,
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.PASS,
                score=1.0 if passed else 0.0,
                passed=passed,
                details={
                    "run_status": run.status.value,
                    "has_exception": run.error_message is not None,
                    "keyword_match_passed": keyword_passed,
                },
            ),
        )
        return results


# ============================================================
# 2. Tool Usage Evaluator
# ============================================================


class ToolUsageEvaluator(BaseEvaluator):
    """Analyze tool calls: count, success rate, redundancy, parameter errors."""

    name = "tool_usage"
    dimension = EvalDimension.TOOL_USAGE

    def evaluate(self, run: RunRecord, spans: list[Span]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        tool_spans = [s for s in spans if s.span_type == SpanType.TOOL_CALL]
        total_calls = len(tool_spans)
        success_calls = sum(1 for s in tool_spans if s.is_success)
        success_rate = (success_calls / total_calls) if total_calls else 1.0

        # Redundant calls: same tool+same arguments called repeatedly
        seen: list[tuple[str, str]] = []
        redundant = 0
        for s in tool_spans:
            try:
                args_key = json.dumps(s.input_data.get("arguments", {}), sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                args_key = str(s.input_data.get("arguments", {}))
            key = (s.name, args_key)
            if key in seen:
                redundant += 1
            else:
                seen.append(key)

        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.TOOL_CALL_COUNT,
                score=total_calls,
                passed=total_calls <= max(10, run.total_steps * 3),  # sanity bound
                details={"by_tool": dict(Counter(s.name for s in tool_spans))},
            )
        )
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.TOOL_SUCCESS_RATE,
                score=round(success_rate, 4),
                passed=success_rate >= 0.8,
                details={"success_calls": success_calls, "total_calls": total_calls},
            )
        )
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.REDUNDANT_CALLS,
                score=redundant,
                passed=redundant == 0,
                details={"redundant_count": redundant, "total_tool_calls": total_calls},
            )
        )
        # Overall tool usage score: weighted combo
        overall = round(0.6 * success_rate + 0.4 * max(0.0, 1.0 - redundant / max(total_calls, 1)), 4)
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                score=overall,
                passed=overall >= 0.7,
                details={"breakdown": f"success_rate({success_rate})*0.6 + redundancy_penalty*0.4"},
            )
        )
        return results


# ============================================================
# 3. Answer Quality Evaluator (Keyword + Heuristic)
# ============================================================


class AnswerQualityEvaluator(BaseEvaluator):
    """Heuristic answer quality: length, structure, keyword coverage, no-refusal."""

    name = "answer_quality_keyword"
    dimension = EvalDimension.ANSWER_QUALITY

    REFUSAL_PATTERNS = [
        "i cannot", "i can't", "sorry", "作为一个人工智能", "我无法", "抱歉",
        "not able to", "i'm sorry", "unable to",
    ]

    def evaluate(self, run: RunRecord, spans: list[Span]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        output = (run.final_output or "").strip()
        expected = (run.expected_output or "").strip()

        if not output:
            return [
                EvaluationResult(
                    run_id=run.run_id,
                    evaluator=self.name,
                    dimension=self.dimension,
                    sub_metric=SubMetric.COMPLETENESS,
                    score=0.0,
                    passed=False,
                    details={"reason": "empty_final_output"},
                )
            ]

        # Length-based completeness
        output_len = len(output)
        expected_len = len(expected) if expected else output_len
        completeness = min(1.0, output_len / max(expected_len, 10))
        # Penalty if way too long (2x expected)
        if expected_len and output_len > expected_len * 2.5:
            completeness *= 0.7

        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.COMPLETENESS,
                score=round(completeness, 4),
                passed=completeness >= 0.5,
                details={
                    "output_chars": output_len,
                    "expected_chars": expected_len,
                },
            )
        )

        # Relevance: check for refusal patterns
        output_lower = output.lower()
        refused = any(p in output_lower for p in self.REFUSAL_PATTERNS)
        relevance = 0.3 if refused else 0.9
        # Boost relevance if expected keywords present
        if expected:
            exp_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", expected.lower()))
            out_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", output_lower))
            if exp_words and out_words:
                overlap = len(exp_words & out_words) / len(exp_words)
                relevance = round(min(1.0, relevance * 0.4 + overlap * 0.6), 4)

        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.RELEVANCE,
                score=relevance,
                passed=relevance >= 0.6,
                details={
                    "refusal_detected": refused,
                    "expected_keyword_overlap": round(
                        (
                            len(set(re.findall(r"\w+", expected.lower())) & set(re.findall(r"\w+", output_lower)))
                            / max(len(set(re.findall(r"\w+", expected.lower()))), 1)
                        )
                        if expected
                        else 0.0,
                        4,
                    ),
                },
            )
        )

        # Correctness: use keyword_match from SuccessRateEvaluator if expected exists
        correctness = 0.5  # default neutral
        if expected:
            exp_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", expected.lower()))
            if exp_words:
                out_words = set(re.findall(r"[a-z0-9\u4e00-\u9fff]{2,}", output_lower))
                correctness = round(len(exp_words & out_words) / len(exp_words), 4)
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.CORRECTNESS,
                score=correctness,
                passed=correctness >= 0.6,
                details={
                    "method": "keyword_overlap_with_expected",
                    "score_range": "0.0~1.0",
                },
            )
        )

        # Overall quality: weighted average of sub-metrics
        sub_scores = [completeness, relevance, correctness]
        overall = round(sum(sub_scores) / len(sub_scores), 4)
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                score=overall,
                passed=overall >= 0.6,
                details={
                    "sub_scores": {
                        "completeness": completeness,
                        "relevance": relevance,
                        "correctness": correctness,
                    },
                    "aggregation": "simple_mean",
                },
            )
        )
        return results


# ============================================================
# 4. Latency Evaluator
# ============================================================


class LatencyEvaluator(BaseEvaluator):
    """Total latency, per-step latency, breakdown by phase (LLM vs Tool)."""

    name = "latency"
    dimension = EvalDimension.LATENCY

    def evaluate(self, run: RunRecord, spans: list[Span]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        total_latency = run.total_latency_ms

        llm_latencies = [s.latency_ms for s in spans if s.span_type == SpanType.LLM_CALL]
        tool_latencies = [s.latency_ms for s in spans if s.span_type == SpanType.TOOL_CALL]

        sum_llm = sum(llm_latencies)
        sum_tool = sum(tool_latencies)
        sum_known = max(total_latency, sum_llm + sum_tool)

        step_latencies = [s.latency_ms for s in spans if s.span_type == SpanType.AGENT_STEP]
        avg_step = (sum(step_latencies) / len(step_latencies)) if step_latencies else 0.0
        if not avg_step:
            # Fallback: total / steps
            avg_step = total_latency / max(run.total_steps, 1)

        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.TOTAL_LATENCY_MS,
                score=total_latency,
                passed=total_latency <= 60_000,  # 60s budget
                details={
                    "unit": "milliseconds",
                    "llm_ms": sum_llm,
                    "tool_ms": sum_tool,
                    "llm_pct": round(sum_llm / sum_known * 100, 2) if sum_known else 0.0,
                    "tool_pct": round(sum_tool / sum_known * 100, 2) if sum_known else 0.0,
                    "num_llm_calls": len(llm_latencies),
                    "num_tool_calls": len(tool_latencies),
                },
            )
        )
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.AVG_STEP_LATENCY_MS,
                score=round(avg_step, 1),
                passed=avg_step <= 10_000,
                details={"unit": "milliseconds", "total_steps": run.total_steps},
            )
        )
        return results


# ============================================================
# 5. Token Cost Evaluator
# ============================================================


class TokenCostEvaluator(BaseEvaluator):
    """Token usage breakdown + total cost + efficiency."""

    name = "token_cost"
    dimension = EvalDimension.TOKEN_COST

    def evaluate(self, run: RunRecord, spans: list[Span]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        t = run.tokens
        cost = run.total_cost

        # Efficiency: ratio of completion tokens that are useful (final answer chars / completion tokens)
        final_chars = len(run.final_output or "")
        completion_tokens = max(t.completion_tokens, 1)
        chars_per_token = final_chars / completion_tokens
        efficiency = round(min(1.0, chars_per_token / 4.0), 4)  # ~4 chars/token expected

        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.PROMPT_TOKENS,
                score=t.prompt_tokens,
                details={"unit": "tokens"},
            )
        )
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.COMPLETION_TOKENS,
                score=t.completion_tokens,
                details={"unit": "tokens"},
            )
        )
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.TOTAL_TOKENS,
                score=t.total_tokens,
                passed=t.total_tokens <= 128_000,
                details={"unit": "tokens"},
            )
        )
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                sub_metric=SubMetric.TOTAL_COST,
                score=round(cost, 6),
                passed=cost <= 1.0,  # under $1 per run
                details={"unit": "USD"},
            )
        )
        # Token efficiency (completion-only usefulness)
        results.append(
            EvaluationResult(
                run_id=run.run_id,
                evaluator=self.name,
                dimension=self.dimension,
                score=efficiency,
                passed=efficiency >= 0.3,
                details={
                    "metric": "token_efficiency",
                    "chars_per_completion_token": round(chars_per_token, 3),
                    "final_answer_chars": final_chars,
                    "interpretation": "higher = more output per token (less fluff)",
                },
            )
        )
        return results


# ============================================================
# Registry helper
# ============================================================


BUILTIN_EVALUATORS: list[type[BaseEvaluator]] = [
    SuccessRateEvaluator,
    ToolUsageEvaluator,
    AnswerQualityEvaluator,
    LatencyEvaluator,
    TokenCostEvaluator,
]


def get_builtin_evaluator_instances(names: list[str] | None = None) -> list[BaseEvaluator]:
    """Instantiate built-in evaluators filtered by name list (None = all).

    Special name ``"llm_judge"`` is handled separately via :class:`LLMJudgeEvaluator`
    which requires an active LLM gateway.
    """
    instances: list[BaseEvaluator] = []
    for cls in BUILTIN_EVALUATORS:
        if names is None or cls.name in names:
            instances.append(cls())
    if names and "llm_judge" in names:
        from agent_eval.evaluation.llm_judge import LLMJudgeEvaluator

        instances.append(LLMJudgeEvaluator())
    return instances
