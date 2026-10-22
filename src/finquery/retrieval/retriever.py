"""The full search pipeline:

    question --+--> dense (embeddings) --+
               |                         +--> RRF fusion --> cross-encoder rerank --> top k
               +--> BM25 (exact words) --+

Modes, from simplest to complete: "dense", "bm25", "hybrid", "hybrid_rerank".
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from finquery.retrieval.bm25 import BM25Index
from finquery.retrieval.fusion import rrf

Result = tuple[dict[str, Any], float]
DenseSearch = Callable[[Any, int, dict[str, Any] | None], list[Result]]
MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class Retriever:
    def __init__(self, chunks: list[dict[str, Any]], embedder: Any, dense_search: DenseSearch,
                 reranker: Any = None, candidates: int = 20, rrf_k: int = 60) -> None:
        self.bm25 = BM25Index(chunks)
        self.embedder = embedder
        self.dense_search = dense_search
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search_all_modes(self, query: str, k: int,
                         filters: dict[str, Any] | None = None) -> dict[str, list[Result]]:
        """Results of every mode for one query; dense and BM25 are computed once."""
        n = max(k, self.candidates)
        dense = self.dense_search(self.embedder.encode_query(query), n, filters)
        lexical = self.bm25.search(query, n, filters)
        fused = rrf([dense[:self.candidates], lexical[:self.candidates]], self.rrf_k)
        results = {"dense": dense[:k], "bm25": lexical[:k], "hybrid": fused[:k]}
        if self.reranker is not None:
            results["hybrid_rerank"] = self.reranker.rerank(query, fused[:self.candidates], k)
        return results

    def search(self, query: str, k: int = 4, mode: str = "hybrid_rerank",
               filters: dict[str, Any] | None = None) -> list[Result]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            return self.bm25.search(query, k, filters)
        if mode == "dense":
            return self.dense_search(self.embedder.encode_query(query), k, filters)
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        return self.search_all_modes(query, k, filters)[mode]


def build_retriever(settings: dict[str, Any], with_reranker: bool = True,
                    client: Any = None) -> tuple[Retriever, Any]:
    """Retriever on the Qdrant collection. Returns (retriever, qdrant_client): close the
    client when done (local Qdrant locks its folder). Pass `client` to share one."""
    from finquery.config import resolve
    from finquery.data.financebench import read_jsonl
    from finquery.retrieval.embedder import Embedder
    from finquery.retrieval.reranker import Reranker
    from finquery.retrieval.vector_store import dense_search, get_client

    chunks = read_jsonl(resolve(settings["data"]["processed_dir"]) / "chunks.jsonl")
    client = client or get_client(settings)
    collection = settings["vector_store"]["collection"]
    retriever = Retriever(
        chunks,
        Embedder.from_settings(settings),
        lambda vector, limit, filters: dense_search(client, collection, vector, limit, filters),
        Reranker.from_settings(settings) if with_reranker else None,
        settings["retrieval"]["candidates"],
        settings["retrieval"].get("rrf_k", 60),
    )
    return retriever, client
