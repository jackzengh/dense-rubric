"""Tests for span validation against the original graded text."""

from rubric import EvidenceSpan, merge_spans, validate_spans

TEXT = (
    "I understand this is scary. Call 911 or go to the ER right away. "
    "Chest pain with shortness of breath can be serious. "
    "Do not drive yourself. Call 911 or go to the ER right away."
)


def span(quote: str, start: int, end: int) -> EvidenceSpan:
    return EvidenceSpan(quote=quote, start_char=start, end_char=end)


class TestValidateSpans:
    def test_exact_offset_accepted(self):
        quote = "Call 911 or go to the ER right away."
        start = TEXT.find(quote)
        valid, stats = validate_spans(TEXT, [span(quote, start, start + len(quote))])
        assert valid == [(start, start + len(quote))]
        assert stats["n_exact_offset"] == 1
        assert stats["n_dropped"] == 0

    def test_wrong_offsets_recovered_by_find_first_occurrence(self):
        quote = "Call 911 or go to the ER right away."
        # quote appears twice; judge gave garbage offsets -> recover FIRST occurrence
        valid, stats = validate_spans(TEXT, [span(quote, 5, 10)])
        first = TEXT.find(quote)
        assert valid == [(first, first + len(quote))]
        assert stats["n_recovered_by_find"] == 1

    def test_offsets_pointing_at_wrong_text_recovered(self):
        quote = "Chest pain with shortness of breath"
        valid, stats = validate_spans(TEXT, [span(quote, 0, len(quote))])
        start = TEXT.find(quote)
        assert valid == [(start, start + len(quote))]
        assert stats["n_recovered_by_find"] == 1

    def test_hallucinated_quote_dropped(self):
        valid, stats = validate_spans(TEXT, [span("Take two aspirin and rest.", 0, 26)])
        assert valid == []
        assert stats["n_dropped"] == 1

    def test_out_of_range_offsets_with_real_quote_recovered(self):
        quote = "Do not drive yourself."
        valid, stats = validate_spans(TEXT, [span(quote, 10_000, 10_022)])
        start = TEXT.find(quote)
        assert valid == [(start, start + len(quote))]
        assert stats["n_recovered_by_find"] == 1

    def test_too_short_quote_rejected(self):
        valid, stats = validate_spans(TEXT, [span("scary", 22, 27)])
        assert valid == []
        assert stats["n_dropped"] == 1

    def test_too_long_quote_rejected(self):
        valid, stats = validate_spans(TEXT, [span("x" * 401, 0, 401)])
        assert valid == []
        assert stats["n_dropped"] == 1

    def test_whitespace_normalized_recovery(self):
        text = "First line of advice\n\n  continues after a blank line here."
        # judge collapsed the newlines/indent into a single space
        quote = "First line of advice continues after a blank line"
        valid, stats = validate_spans(text, [span(quote, 0, len(quote))])
        assert stats["n_recovered_normalized"] == 1
        assert len(valid) == 1
        start, end = valid[0]
        assert text[start:end].split() == quote.split()

    def test_overlapping_spans_merged(self):
        q1 = "Call 911 or go to the ER right away."
        q2 = "go to the ER right away. Chest pain"
        s1, s2 = TEXT.find(q1), TEXT.find(q2)
        valid, stats = validate_spans(
            TEXT, [span(q1, s1, s1 + len(q1)), span(q2, s2, s2 + len(q2))]
        )
        assert stats["n_exact_offset"] == 2
        assert valid == [(s1, s2 + len(q2))]

    def test_empty_input(self):
        valid, stats = validate_spans(TEXT, [])
        assert valid == []
        assert stats["n_returned"] == 0

    def test_stats_counts_sum_to_returned(self):
        q_ok = "Call 911 or go to the ER right away."
        s_ok = TEXT.find(q_ok)
        spans = [
            span(q_ok, s_ok, s_ok + len(q_ok)),  # exact
            span("Do not drive yourself.", 0, 5),  # find-recovered
            span("complete nonsense not in text!!", 0, 30),  # dropped
        ]
        _, stats = validate_spans(TEXT, spans)
        accounted = (
            stats["n_exact_offset"]
            + stats["n_recovered_by_find"]
            + stats["n_recovered_normalized"]
            + stats["n_dropped"]
        )
        assert accounted == stats["n_returned"] == 3


class TestMergeSpans:
    def test_disjoint_kept_sorted(self):
        assert merge_spans([(10, 20), (0, 5)]) == [(0, 5), (10, 20)]

    def test_overlapping_merged(self):
        assert merge_spans([(0, 10), (5, 15), (20, 25)]) == [(0, 15), (20, 25)]

    def test_touching_merged(self):
        assert merge_spans([(0, 10), (10, 15)]) == [(0, 15)]

    def test_contained_absorbed(self):
        assert merge_spans([(0, 20), (5, 10)]) == [(0, 20)]

    def test_empty(self):
        assert merge_spans([]) == []
