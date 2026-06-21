"""Validation of judge-returned evidence spans against the original graded text.

The judge promises `text[start_char:end_char] == quote`, but LLMs routinely get
offsets wrong even when the quote itself is correct. Validation therefore runs a
ladder of checks, from strictest to most forgiving:

1. exact offset match        -> trust the judge's offsets
2. text.find(quote)          -> quote is verbatim but offsets were wrong; use the
                                first occurrence (spec: first occurrence wins)
3. whitespace-normalized find -> quote differs only in whitespace (judges often
                                collapse newlines); recover offsets via an offset map
4. drop the span             -> the quote is not in the text; the criterion's
                                verdict/score are kept, only the span is discarded
"""

import re

from rubric.autograders.schemas import EvidenceSpan

# Spans shorter than this are too ambiguous to localize ("yes", "the") and spans
# longer than this stop being evidence and start being the whole answer.
DEFAULT_MIN_QUOTE_CHARS = 10
DEFAULT_MAX_QUOTE_CHARS = 400

_WHITESPACE_RUN = re.compile(r"\s+")


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching (start, end) intervals into a sorted, disjoint list."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _normalize_with_offset_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces, keeping a map from each normalized
    character back to its offset in the original text."""
    normalized_chars: list[str] = []
    offset_map: list[int] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            # emit one space for the whole whitespace run
            normalized_chars.append(" ")
            offset_map.append(i)
            while i < len(text) and text[i].isspace():
                i += 1
        else:
            normalized_chars.append(text[i])
            offset_map.append(i)
            i += 1
    return "".join(normalized_chars), offset_map


def _find_whitespace_normalized(text: str, quote: str) -> tuple[int, int] | None:
    """Locate `quote` in `text` ignoring whitespace differences. Returns original-text
    offsets of the first match, or None."""
    norm_text, offset_map = _normalize_with_offset_map(text)
    norm_quote = _WHITESPACE_RUN.sub(" ", quote).strip()
    if not norm_quote:
        return None
    idx = norm_text.find(norm_quote)
    if idx == -1:
        return None
    start = offset_map[idx]
    last_norm_idx = idx + len(norm_quote) - 1
    # the last normalized char maps to the start of its original run; the span ends
    # one past that original character
    end = offset_map[last_norm_idx] + 1
    return (start, end)


def validate_spans(
    text: str,
    spans: list[EvidenceSpan],
    min_quote_chars: int = DEFAULT_MIN_QUOTE_CHARS,
    max_quote_chars: int = DEFAULT_MAX_QUOTE_CHARS,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Validate judge-returned spans against the original text.

    Returns:
        (valid_spans, stats) where valid_spans is a merged, disjoint list of
        (start, end) character intervals and stats counts each validation outcome:
        n_returned, n_exact_offset, n_recovered_by_find, n_recovered_normalized,
        n_dropped.
    """
    stats = {
        "n_returned": len(spans),
        "n_exact_offset": 0,
        "n_recovered_by_find": 0,
        "n_recovered_normalized": 0,
        "n_dropped": 0,
    }
    valid: list[tuple[int, int]] = []

    for span in spans:
        quote = span.quote
        if not (min_quote_chars <= len(quote) <= max_quote_chars):
            stats["n_dropped"] += 1
            continue

        start, end = span.start_char, span.end_char
        if 0 <= start < end <= len(text) and text[start:end] == quote:
            valid.append((start, end))
            stats["n_exact_offset"] += 1
            continue

        idx = text.find(quote)
        if idx != -1:
            valid.append((idx, idx + len(quote)))
            stats["n_recovered_by_find"] += 1
            continue

        recovered = _find_whitespace_normalized(text, quote)
        if recovered is not None:
            valid.append(recovered)
            stats["n_recovered_normalized"] += 1
            continue

        stats["n_dropped"] += 1

    return merge_spans(valid), stats
