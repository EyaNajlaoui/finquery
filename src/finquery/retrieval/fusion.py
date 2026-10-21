"""Reciprocal Rank Fusion (RRF).

Dense scores (cosine, ~0.7-0.8) and BM25 scores (~5-20) are not on the same scale,
so they cannot be added. RRF only uses ranks: score = sum over lists of 1 / (k + rank).
A chunk ranked well by both searches beats a chunk ranked first by only one.
"""
from __future__ import annotations

from typing import Any

Result = tuple[dict[str, Any], float]


def rrf(result_lists: list[list[Result]], k: int = 60) -> list[Result]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, (chunk, _) in enumerate(results, start=1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1 / (k + rank)
            chunks.setdefault(cid, chunk)
    order = sorted(scores, key=scores.get, reverse=True)
    return [(chunks[cid], scores[cid]) for cid in order]
