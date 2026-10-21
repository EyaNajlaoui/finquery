"""Brute-force dense search in numpy, used for experiments (e.g. comparing chunk
sizes) without touching the Qdrant collection. 5,000 x 384 vectors fit easily in memory."""
from __future__ import annotations

from typing import Any

import numpy as np

from finquery.retrieval.filters import matches


class NumpyDenseIndex:
    def __init__(self, chunks: list[dict[str, Any]], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk expected")
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, vector: Any, limit: int = 20,
               filters: dict[str, Any] | None = None) -> list[tuple[dict[str, Any], float]]:
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)  # vectors are normalized
        if filters:
            allowed = np.array([matches(c, filters) for c in self.chunks])
            scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores)[:limit]
        return [(self.chunks[i], float(scores[i])) for i in order if np.isfinite(scores[i])]
