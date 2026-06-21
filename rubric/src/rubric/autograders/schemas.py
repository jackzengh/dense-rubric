"""Pydantic schemas for LLM output validation.

These models define the expected output structure for each autograder type.
Users can access `.model_json_schema()` on these models to enable constrained
decoding in their LLM clients.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PerCriterionOutput(BaseModel):
    """Expected output for PerCriterionGrader.

    Evaluates a single criterion and returns MET/UNMET verdict with explanation.

    Example:
        >>> output = PerCriterionOutput(
        ...     explanation="The response correctly identifies the diagnosis.",
        ...     criterion_status="MET"
        ... )
    """

    explanation: str = Field(
        description="Brief explanation of whether the criterion is present (MET) or \
        absent (UNMET) in the response.",
    )
    criterion_status: Literal["MET", "UNMET"] = Field(
        description="Whether the criterion is present (MET) or absent (UNMET) in the response."
    )


class EvidenceSpan(BaseModel):
    """A verbatim quote from the graded response with exact character offsets.

    The contract is `response[start_char:end_char] == quote`. Offsets are 0-based,
    start inclusive, end exclusive. Spans that violate the contract are dropped by
    validation while the criterion's verdict and score are kept.
    """

    quote: str = Field(
        description="Verbatim, character-for-character substring copied from the response, \
        including punctuation, casing, and whitespace."
    )
    start_char: int = Field(
        description="0-based inclusive character offset of the quote's first character in the \
        response, such that response[start_char:end_char] == quote."
    )
    end_char: int = Field(
        description="0-based exclusive character offset of the quote's end in the response."
    )


class PerCriterionSpanOutput(BaseModel):
    """Expected output for SpanPerCriterionGrader.

    Same MET/UNMET verdict as PerCriterionOutput, plus evidence spans locating where
    in the response the criterion is satisfied (positive criteria) or where the error
    is committed (negative criteria).

    Example:
        >>> output = PerCriterionSpanOutput(
        ...     explanation="The response recommends seeing a doctor within 24 hours.",
        ...     criterion_status="MET",
        ...     localizable=True,
        ...     evidence=[EvidenceSpan(quote="see a doctor within 24 hours",
        ...                            start_char=112, end_char=140)],
        ... )
    """

    explanation: str = Field(
        description="Brief explanation of whether the criterion is present (MET) or \
        absent (UNMET) in the response.",
    )
    criterion_status: Literal["MET", "UNMET"] = Field(
        description="Whether the criterion is present (MET) or absent (UNMET) in the response."
    )
    localizable: bool = Field(
        description="False when the criterion is global and cannot be tied to specific text \
        (overall tone, completeness, absence of content). When False, evidence must be empty."
    )
    evidence: list[EvidenceSpan] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 3 verbatim evidence spans. Empty when criterion_status is UNMET for \
        positive criteria, when no error text exists for negative criteria, or when the \
        criterion is not localizable.",
    )


class CriterionEvaluation(BaseModel):
    """A single criterion evaluation within a one-shot response.

    Used by OneShotOutput to represent each criterion's verdict.
    """

    criterion_idx: int = Field(
        description="The 0-based index of the criterion being evaluated.",
    )
    explanation: str = Field(
        description="Brief explanation of whether the criterion is present (MET) or \
        absent (UNMET) in the response."
    )
    criterion_status: Literal["MET", "UNMET"] = Field(
        description="Whether the criterion is present (MET) or absent (UNMET) in the response."
    )


class OneShotOutput(BaseModel):
    """Expected output for PerCriterionOneShotGrader.

    Evaluates all criteria in a single LLM call.

    Example:
        >>> output = OneShotOutput(criteria_evaluations=[
        ...     CriterionEvaluation(criterion_idx=0, explanation="...", criterion_status="MET"),
        ...     CriterionEvaluation(criterion_idx=1, explanation="...", criterion_status="UNMET")
        ... ])
    """

    criteria_evaluations: list[CriterionEvaluation] = Field(
        description="List of evaluations for each criterion.", min_length=1
    )


class RubricAsJudgeOutput(BaseModel):
    """Expected output for RubricAsJudgeGrader.

    Returns a single holistic score from 0-100.

    Example:
        >>> output = RubricAsJudgeOutput(explanation="...", overall_score=85.0)
    """

    explanation: str = Field(
        description="Brief explanation of the holistic score from 0-100 representing overall \
        rubric satisfaction.",
    )
    overall_score: float = Field(
        description="Holistic score from 0-100 representing overall rubric satisfaction.",
    )
