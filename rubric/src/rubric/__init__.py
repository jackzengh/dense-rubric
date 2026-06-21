from rubric.autograders.schemas import (
    CriterionEvaluation,
    EvidenceSpan,
    OneShotOutput,
    PerCriterionOutput,
    PerCriterionSpanOutput,
    RubricAsJudgeOutput,
)
from rubric.rubric import Rubric
from rubric.span_validation import merge_spans, validate_spans
from rubric.types import (
    Criterion,
    CriterionReport,
    EvaluationReport,
    OneShotGenerateFn,
    PerCriterionGenerateFn,
    RubricAsJudgeGenerateFn,
    SpanCriterionReport,
    SpanPerCriterionGenerateFn,
)
from rubric.utils import (
    default_oneshot_generate_fn,
    default_per_criterion_generate_fn,
    default_rubric_as_judge_generate_fn,
)

__version__ = "2.3.0.dev0"
__all__ = [
    "Criterion",
    "CriterionEvaluation",
    "CriterionReport",
    "EvaluationReport",
    "EvidenceSpan",
    "Rubric",
    "SpanCriterionReport",
    "default_oneshot_generate_fn",
    "default_per_criterion_generate_fn",
    "default_rubric_as_judge_generate_fn",
    "merge_spans",
    "validate_spans",
    "PerCriterionGenerateFn",
    "OneShotGenerateFn",
    "RubricAsJudgeGenerateFn",
    "SpanPerCriterionGenerateFn",
    "OneShotOutput",
    "PerCriterionOutput",
    "PerCriterionSpanOutput",
    "RubricAsJudgeOutput",
]
__name__ = "rubric"
__author__ = "The LLM Data Company"
