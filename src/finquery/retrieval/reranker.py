"""Cross-encoder reranking.

The embedding model encodes the question and the chunk separately; a cross-encoder
reads them together, so it sees which words of the chunk answer which words of the
question. Too slow for 5,000 chunks, fine for the 20 candidates of the hybrid search.
Its scores are only used to sort: they are not probabilities, so no threshold on them.
"""
from __future__ import annotations

from typing import Any

from finquery.ingestion.chunker import text_for_embedding

Result = tuple[dict[str, Any], float]


class Reranker:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder  # heavy import, done lazily

        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> Reranker:
        cfg = settings["reranker"]
        return cls(cfg["model"], cfg.get("device", "cpu"), cfg.get("batch_size", 16))

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                    show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, results: list[Result], limit: int | None = None) -> list[Result]:
        if not results:
            return []
        scores = self.score(query, [text_for_embedding(chunk) for chunk, _ in results])
        order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        return [(results[i][0], scores[i]) for i in order[:limit]]
