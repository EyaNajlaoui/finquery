"""Lexical search with BM25.

Dense embeddings are weak on exact strings: tickers, line items
("days payable outstanding"), and above all numbers. BM25 matches them exactly.
The index is rebuilt in memory from chunks.jsonl (a few seconds for 5k chunks),
so there is nothing extra to store or to keep in sync.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from finquery.evaluation.metrics import tokens
from finquery.ingestion.chunker import text_for_embedding
from finquery.retrieval.filters import matches

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what which with how much many did does do during".split())


def tokenize(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(text_for_embedding(c)) for c in chunks]) if chunks else None

    def search(self, query: str, limit: int = 20,
               filters: dict[str, Any] | None = None) -> list[tuple[dict[str, Any], float]]:
        if self.bm25 is None:  # nothing indexed yet (no library, no upload)
            return []
        scores = self.bm25.get_scores(tokenize(query))
        if filters:
            allowed = np.array([matches(c, filters) for c in self.chunks])
            scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores)[:limit]
        return [(self.chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i]) and scores[i] > 0]
