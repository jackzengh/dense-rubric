"""Tests for SpanPerCriterionGrader using mocked generate functions (no network)."""

import re

import pytest

from rubric import (
    Criterion,
    EvidenceSpan,
    PerCriterionSpanOutput,
    SpanCriterionReport,
)
from rubric.autograders import SpanPerCriterionGrader

RESPONSE = (
    "You should see a doctor within 24 hours for these symptoms. "
    "In the meantime, drink plenty of fluids and rest."
)


def make_generate_fn(outputs_by_requirement: dict[str, PerCriterionSpanOutput]):
    """Build a mock generate_fn that picks its canned output by the requirement inside
    the prompt's <criterion> block (matching the full prompt would also match the
    response text, which may contain the same words)."""

    async def generate_fn(system_prompt: str, user_prompt: str, **kwargs):
        match = re.search(r"<criterion>\n(.*?)\n</criterion>", user_prompt, re.DOTALL)
        requirement = match.group(1).strip()
        for key, output in outputs_by_requirement.items():
            if key in requirement:
                return output
        raise AssertionError(f"No canned output matches criterion: {requirement}")

    return generate_fn


@pytest.fixture
def criteria() -> list[Criterion]:
    return [
        Criterion(weight=5.0, requirement="Advises seeing a doctor within 24 hours"),
        Criterion(weight=2.0, requirement="Recommends hydration"),
        Criterion(weight=-4.0, requirement="Recommends antibiotics without a prescription"),
    ]


class TestSpanGrading:
    @pytest.mark.asyncio
    async def test_valid_spans_attached_and_scored(self, criteria):
        quote = "see a doctor within 24 hours"
        start = RESPONSE.find(quote)
        generate_fn = make_generate_fn(
            {
                "doctor within 24 hours": PerCriterionSpanOutput(
                    explanation="Advises a doctor visit within 24 hours.",
                    criterion_status="MET",
                    localizable=True,
                    evidence=[
                        EvidenceSpan(quote=quote, start_char=start, end_char=start + len(quote))
                    ],
                ),
                "hydration": PerCriterionSpanOutput(
                    explanation="Recommends fluids.",
                    criterion_status="MET",
                    localizable=True,
                    evidence=[
                        EvidenceSpan(quote="drink plenty of fluids", start_char=0, end_char=5)
                    ],  # wrong offsets -> recovered via find
                ),
                "antibiotics": PerCriterionSpanOutput(
                    explanation="No antibiotics mentioned.",
                    criterion_status="UNMET",
                    localizable=True,
                    evidence=[],
                ),
            }
        )
        grader = SpanPerCriterionGrader(generate_fn=generate_fn)
        result = await grader.grade(RESPONSE, criteria)

        # scoring identical to PerCriterionGrader: (5 + 2) / (5 + 2) = 1.0
        assert result.score == 1.0
        assert result.raw_score == 7.0
        assert all(isinstance(r, SpanCriterionReport) for r in result.report)

        doctor = next(r for r in result.report if "doctor" in r.requirement)
        assert doctor.valid_spans == [(start, start + len(quote))]
        assert doctor.span_stats["n_exact_offset"] == 1

        hydration = next(r for r in result.report if "hydration" in r.requirement)
        fluids_start = RESPONSE.find("drink plenty of fluids")
        assert hydration.valid_spans == [(fluids_start, fluids_start + len("drink plenty of fluids"))]
        assert hydration.span_stats["n_recovered_by_find"] == 1

        antibiotics = next(r for r in result.report if "antibiotics" in r.requirement)
        assert antibiotics.valid_spans == []

    @pytest.mark.asyncio
    async def test_invalid_span_dropped_but_score_kept(self, criteria):
        generate_fn = make_generate_fn(
            {
                "doctor within 24 hours": PerCriterionSpanOutput(
                    explanation="Met.",
                    criterion_status="MET",
                    localizable=True,
                    evidence=[
                        EvidenceSpan(
                            quote="this text is nowhere in the response",
                            start_char=0,
                            end_char=37,
                        )
                    ],
                ),
                "hydration": PerCriterionSpanOutput(
                    explanation="Unmet.", criterion_status="UNMET", localizable=True, evidence=[]
                ),
                "antibiotics": PerCriterionSpanOutput(
                    explanation="Unmet.", criterion_status="UNMET", localizable=True, evidence=[]
                ),
            }
        )
        grader = SpanPerCriterionGrader(generate_fn=generate_fn)
        result = await grader.grade(RESPONSE, criteria)

        doctor = next(r for r in result.report if "doctor" in r.requirement)
        # span dropped...
        assert doctor.valid_spans == []
        assert doctor.span_stats["n_dropped"] == 1
        # ...but the MET verdict still earns its 5 points: 5 / 7
        assert result.raw_score == 5.0

    @pytest.mark.asyncio
    async def test_non_localizable_criterion_contributes_no_spans(self, criteria):
        quote = "see a doctor within 24 hours"
        start = RESPONSE.find(quote)
        generate_fn = make_generate_fn(
            {
                "doctor within 24 hours": PerCriterionSpanOutput(
                    explanation="Global criterion.",
                    criterion_status="MET",
                    localizable=False,
                    # judge returned evidence anyway; localizable=False must discard it
                    evidence=[
                        EvidenceSpan(quote=quote, start_char=start, end_char=start + len(quote))
                    ],
                ),
                "hydration": PerCriterionSpanOutput(
                    explanation="Unmet.", criterion_status="UNMET", localizable=True, evidence=[]
                ),
                "antibiotics": PerCriterionSpanOutput(
                    explanation="Unmet.", criterion_status="UNMET", localizable=True, evidence=[]
                ),
            }
        )
        grader = SpanPerCriterionGrader(generate_fn=generate_fn)
        result = await grader.grade(RESPONSE, criteria)

        doctor = next(r for r in result.report if "doctor" in r.requirement)
        assert doctor.localizable is False
        assert doctor.valid_spans == []
        assert result.raw_score == 5.0  # verdict still counts

    @pytest.mark.asyncio
    async def test_negative_criterion_met_failure_span(self, criteria):
        bad_response = "Just take some leftover amoxicillin from your last prescription."
        quote = "take some leftover amoxicillin"
        start = bad_response.find(quote)
        generate_fn = make_generate_fn(
            {
                "doctor within 24 hours": PerCriterionSpanOutput(
                    explanation="Unmet.", criterion_status="UNMET", localizable=True, evidence=[]
                ),
                "hydration": PerCriterionSpanOutput(
                    explanation="Unmet.", criterion_status="UNMET", localizable=True, evidence=[]
                ),
                "antibiotics": PerCriterionSpanOutput(
                    explanation="Recommends unprescribed antibiotics.",
                    criterion_status="MET",
                    localizable=True,
                    evidence=[
                        EvidenceSpan(quote=quote, start_char=start, end_char=start + len(quote))
                    ],
                ),
            }
        )
        grader = SpanPerCriterionGrader(generate_fn=generate_fn)
        result = await grader.grade(bad_response, criteria)

        antibiotics = next(r for r in result.report if "antibiotics" in r.requirement)
        assert antibiotics.verdict == "MET"
        assert antibiotics.valid_spans == [(start, start + len(quote))]
        # raw score: 0 + 0 + (-4) = -4; normalized clamps to 0
        assert result.raw_score == -4.0
        assert result.score == 0.0
