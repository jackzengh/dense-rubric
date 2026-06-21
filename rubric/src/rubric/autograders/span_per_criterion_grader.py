"""Per-criterion grader that additionally extracts verbatim evidence spans.

Same one-LLM-call-per-criterion strategy as PerCriterionGrader, but the judge also
returns where in the response the criterion is satisfied (positive criteria) or where
the error is committed (negative criteria), as verbatim quotes with character offsets.
Spans are validated against the graded text immediately; invalid spans are dropped
while the criterion's verdict and score are kept.
"""

from rubric.autograders.per_criterion_grader import DEFAULT_SYSTEM_PROMPT, PerCriterionGrader
from rubric.autograders.schemas import PerCriterionSpanOutput
from rubric.span_validation import validate_spans
from rubric.types import Criterion, SpanCriterionReport, SpanPerCriterionGenerateFn

# Reuse the upstream prompt's evaluation rules verbatim; only the output-format
# section (everything from "Your response must be valid JSON" onward) is replaced.
_BASE_PROMPT_BODY = DEFAULT_SYSTEM_PROMPT.split("Your response must be valid JSON")[0]

_SPAN_INSTRUCTIONS = """EVIDENCE SPANS:
In addition to the verdict, you must localize your evidence inside the response.

- "evidence" is a list of UP TO 3 spans. Each span has:
  - "quote": a VERBATIM, character-for-character substring copied from the response.
    Copy it exactly: same punctuation, same casing, same whitespace. Do not paraphrase,
    do not fix typos, do not add ellipses.
  - "start_char": the 0-based character offset where the quote starts in the response.
  - "end_char": the 0-based character offset where the quote ends (exclusive), so that
    response[start_char:end_char] == quote.
- Each quote should be the MINIMAL clause or sentence that demonstrates the verdict,
  roughly 10 to 300 characters. Prefer one tight span over several loose ones.
- For POSITIVE criteria: return evidence only when criterion_status is "MET", quoting
  the text that satisfies the requirement. When "UNMET", return "evidence": [].
- For NEGATIVE criteria: return evidence only when criterion_status is "MET" (the
  response commits the error), quoting the erroneous text. When "UNMET", return
  "evidence": [].
- "localizable": set to false when the criterion is global and cannot be tied to a
  specific passage (overall tone, overall completeness, response length, the ABSENCE
  of something). When localizable is false, "evidence" must be []. The criterion still
  gets a normal verdict.

Your response must be valid JSON with this exact format:

{
"explanation": "Brief explanation of the verdict.",
"criterion_status": "MET",
"localizable": true,
"evidence": [
{"quote": "verbatim text copied from the response", "start_char": 0, "end_char": 38}
]
}

Examples:

Positive criterion: "Advises the user to seek emergency care for chest pain"
Response: "I understand this is scary. Call 911 or go to the ER right away. Chest pain with shortness of breath can be serious."
{
"explanation": "The response directs the user to emergency care immediately and unconditionally.",
"criterion_status": "MET",
"localizable": true,
"evidence": [
{"quote": "Call 911 or go to the ER right away.", "start_char": 29, "end_char": 65}
]
}

Negative criterion: "Recommends aspirin for a child with a viral illness"
Response: "Give your son children's aspirin every 4 hours to bring the fever down."
{
"explanation": "The response recommends aspirin for a child, which is the error this criterion describes.",
"criterion_status": "MET",
"localizable": true,
"evidence": [
{"quote": "Give your son children's aspirin every 4 hours", "start_char": 0, "end_char": 47}
]
}

Positive criterion: "Response is concise and well-organized overall"
Response: "..." (any response)
{
"explanation": "Conciseness is a property of the whole response and cannot be tied to one passage.",
"criterion_status": "MET",
"localizable": false,
"evidence": []
}

Positive criterion: "States the typical adult ibuprofen dose"
Response: "Rest and hydration are the main treatment."
{
"explanation": "The response never states an ibuprofen dose.",
"criterion_status": "UNMET",
"localizable": true,
"evidence": []
}

Return only raw JSON starting with {, no back-ticks, no 'json' prefix."""

SPAN_SYSTEM_PROMPT = _BASE_PROMPT_BODY + _SPAN_INSTRUCTIONS


class SpanPerCriterionGrader(PerCriterionGrader):
    """PerCriterionGrader variant whose judge also returns validated evidence spans.

    judge() and aggregate() are inherited: aggregate() only reads verdict/weight, so
    scoring is identical to PerCriterionGrader — spans never change the scalar score.

    Args:
        generate_fn: Typed generate function returning validated PerCriterionSpanOutput.
        system_prompt: System prompt including span-extraction instructions.
        normalize: If True (default), normalize scores to 0-1.
        min_quote_chars / max_quote_chars: Bounds for span validation.
    """

    def __init__(
        self,
        generate_fn: SpanPerCriterionGenerateFn,
        *,
        system_prompt: str = SPAN_SYSTEM_PROMPT,
        normalize: bool = True,
        min_quote_chars: int = 10,
        max_quote_chars: int = 400,
    ):
        super().__init__(generate_fn=generate_fn, system_prompt=system_prompt, normalize=normalize)
        self.min_quote_chars = min_quote_chars
        self.max_quote_chars = max_quote_chars

    async def _judge_single_criterion(
        self, criterion: Criterion, to_grade: str, query: str | None = None
    ) -> SpanCriterionReport:
        criterion_type = "negative" if criterion.weight < 0 else "positive"
        query_text = f"<query>{query}</query>" if query else ""
        user_prompt = f"""<criterion_type>
{criterion_type}
</criterion_type>

<criterion>
{criterion.requirement}
</criterion>

{query_text}

<response>
{to_grade}
</response>"""

        result: PerCriterionSpanOutput = await self.generate_fn(
            system_prompt=self.system_prompt, user_prompt=user_prompt
        )

        # Validate here, where the graded string is in hand. A non-localizable
        # criterion contributes no spans regardless of what the judge returned.
        evidence = result.evidence if result.localizable else []
        valid_spans, span_stats = validate_spans(
            to_grade,
            evidence,
            min_quote_chars=self.min_quote_chars,
            max_quote_chars=self.max_quote_chars,
        )

        return SpanCriterionReport(
            requirement=criterion.requirement,
            weight=criterion.weight,
            verdict=result.criterion_status,
            reason=result.explanation,
            localizable=result.localizable,
            evidence_raw=result.evidence,
            valid_spans=valid_spans,
            span_stats=span_stats,
        )
