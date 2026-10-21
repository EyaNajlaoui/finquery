"""Retrieval and answer metrics, computed by code (no LLM)."""
from __future__ import annotations

import re
from collections.abc import Sequence

_NUMBER = re.compile(r"-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\$?\d+(?:\.\d+)?")


def recall_at_k(retrieved_pages: Sequence[int], relevant_pages: Sequence[int], k: int) -> float:
    """Share of relevant pages found among the first k retrieved chunks' pages."""
    if not relevant_pages:
        raise ValueError("relevant_pages is empty")
    top = set(retrieved_pages[:k])
    return sum(1 for p in set(relevant_pages) if p in top) / len(set(relevant_pages))


def reciprocal_rank(retrieved_pages: Sequence[int], relevant_pages: Sequence[int], k: int) -> float:
    """1 / rank of the first relevant chunk in the top k, 0 if none."""
    relevant = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in relevant:
            return 1 / rank
    return 0.0


def extract_numbers(text: str) -> list[float]:
    """'$1,234.5 million', '15.98%' and '-3' -> [1234.5, 15.98, -3.0]."""
    return [float(n.replace("$", "").replace(",", "")) for n in _NUMBER.findall(text)]


def contains_number(text: str, expected: float, rel_tol: float = 0.01) -> bool:
    """True if the text contains a number within rel_tol (relative) of expected,
    also accepting the percentage form of a ratio (0.62 vs 62%)."""
    targets = [expected] + ([expected * 100] if abs(expected) < 1 else [])
    return any(abs(n - t) <= max(abs(t) * rel_tol, 1e-9)
               for n in extract_numbers(text) for t in targets)
