"""Build the question-answering systems from settings.yaml.

- Answerer (step 4): one search, then one answer
- Agent (step 5): the LLM calls search and calculator tools
Both can share one retriever (and one Qdrant client: local Qdrant locks its folder).
"""
from __future__ import annotations

from typing import Any

from finquery.config import resolve
from finquery.data.financebench import read_jsonl
from finquery.generation.answerer import Answerer
from finquery.generation.context import ContextBuilder
from finquery.generation.llm import GroqLLM
from finquery.retrieval.retriever import Retriever, build_retriever


def load_pages(settings: dict[str, Any]) -> dict[tuple[str, int], str]:
    pages = {}
    for path in (resolve(settings["data"]["processed_dir"]) / "pages").glob("*.jsonl"):
        for p in read_jsonl(path):
            pages[p["doc_name"], p["page"]] = p["text"]
    return pages


def make_answerer(settings: dict[str, Any], retriever: Retriever, pages: dict, llm: Any,
                  context_mode: str | None = None) -> Answerer:
    gen = settings["generation"]
    context = ContextBuilder(retriever.bm25.chunks, pages, mode=context_mode or gen["context_mode"],
                             neighbours=gen.get("neighbours", 1),
                             max_passages=settings["retrieval"]["top_k"])
    return Answerer(retriever, context, llm, k=settings["retrieval"]["top_k"],
                    mode=gen.get("retrieval_mode", "hybrid_rerank"))


def make_agent(settings: dict[str, Any], retriever: Retriever, pages: dict, llm: Any) -> Any:
    from finquery.agent.agent import Agent

    cfg, gen = settings["agent"], settings["generation"]
    context = ContextBuilder(retriever.bm25.chunks, pages, mode=gen["context_mode"],
                             neighbours=gen.get("neighbours", 1),
                             max_passages=cfg.get("passages_per_search", 3))
    catalog = read_jsonl(resolve(settings["data"]["documents_file"]))
    return Agent(retriever, context, llm, catalog, max_steps=cfg.get("max_steps", 6),
                 mode=gen.get("retrieval_mode", "hybrid_rerank"),
                 max_chars_per_search=cfg.get("max_chars_per_search", 4000),
                 max_total_chars=cfg.get("max_total_chars", 16000))


def build_answerer(settings: dict[str, Any], context_mode: str | None = None,
                   llm: Any = None) -> tuple[Answerer, Any]:
    """Returns (answerer, qdrant_client); close the client when done."""
    retriever, client = build_retriever(settings)
    answerer = make_answerer(settings, retriever, load_pages(settings),
                             llm or GroqLLM.from_settings(settings), context_mode)
    return answerer, client


def build_agent(settings: dict[str, Any], llm: Any = None) -> tuple[Any, Any]:
    """Returns (agent, qdrant_client); close the client when done."""
    retriever, client = build_retriever(settings)
    agent = make_agent(settings, retriever, load_pages(settings), llm or GroqLLM.from_settings(settings))
    return agent, client


class FinQuerySystem:
    """Everything the app needs: models loaded once, the benchmark filings plus the user's
    uploads, and both answering modes. Uploads have their own Qdrant collection."""

    def __init__(self, settings: dict[str, Any], client: Any = None) -> None:
        from finquery.ingestion.uploads import UploadStore
        from finquery.retrieval.embedder import Embedder
        from finquery.retrieval.reranker import Reranker
        from finquery.retrieval.vector_store import get_client

        self.settings = settings
        self.client = client or get_client(settings)
        self.embedder = Embedder.from_settings(settings)
        self.reranker = Reranker.from_settings(settings)
        self.llm = GroqLLM.from_settings(settings)
        self.small_llm = GroqLLM.from_settings(settings, role="judge_model", fallback=False)
        data, store = settings["data"], settings["vector_store"]
        self.library_collection = store["collection"]
        self.upload_collection = settings["uploads"]["collection"]
        processed = resolve(data["processed_dir"])
        chunks_file = processed / "chunks.jsonl"
        self.library_chunks = read_jsonl(chunks_file) if chunks_file.exists() else []
        self.library_pages = load_pages(settings)
        docs_file = resolve(data["documents_file"])
        self.library_catalog = [{**d, "source": "library"} for d in read_jsonl(docs_file)] \
            if docs_file.exists() else []
        self.uploads = UploadStore(resolve(settings["uploads"]["dir"]),
                                   settings["chunking"]["max_words"], settings["chunking"]["overlap_words"])
        self.refresh()

    # --- search over both collections ------------------------------------------------------
    def _dense(self, vector: Any, limit: int, filters: dict[str, Any] | None) -> list:
        from finquery.retrieval.vector_store import dense_search

        results = []
        for name in (self.library_collection, self.upload_collection):
            if self.client.collection_exists(name):
                results += dense_search(self.client, name, vector, limit, filters)
        return sorted(results, key=lambda r: r[1], reverse=True)[:limit]  # same model: comparable

    def refresh(self) -> None:
        """Rebuild BM25 and the context builders after an upload or a removal (a few seconds)."""
        chunks = self.library_chunks + self.uploads.chunks()
        pages = {**self.library_pages, **self.uploads.pages()}
        cfg = self.settings
        self.retriever = Retriever(chunks, self.embedder, self._dense, self.reranker,
                                   cfg["retrieval"]["candidates"], cfg["retrieval"].get("rrf_k", 60))
        self.answerer = make_answerer(cfg, self.retriever, pages, self.llm)
        self.agent = make_agent(cfg, self.retriever, pages, self.llm)
        self.agent.catalog = self.catalog()

    def catalog(self) -> list[dict[str, Any]]:
        return [{**d, "source": "upload"} for d in self.uploads.catalog()] + self.library_catalog

    # --- actions ---------------------------------------------------------------------------
    def ask(self, question: str, agent: bool = True, doc_names: list[str] | None = None) -> dict[str, Any]:
        filters = {"doc_name": doc_names} if doc_names else None
        try:
            result = (self.agent if agent else self.answerer).answer(question, filters)
        except Exception as error:
            self._log({"event": "error", "mode": "agent" if agent else "fast", "question": question,
                       "error": f"{type(error).__name__}: {str(error)[:300]}"})
            raise
        self._log({"event": "answer", "mode": "agent" if agent else "fast", "question": question,
                   "documents": len(doc_names or []), "model": result["model"], "seconds": result["seconds"],
                   "tokens": sum(result["usage"].values()), "pages_read": len(result["passages"]),
                   "citations": len(result["citations"]),
                   "invalid_citations": len(result["invalid_citations"]),
                   "refusal": result["refusal"], "steps": result.get("steps")})
        return result

    def _log(self, record: dict[str, Any]) -> None:
        """One JSON line per event in data/logs/queries.jsonl: latency, cost and quality signals
        (refusals, invalid citations) of real use, to watch the app after deployment."""
        import json
        from datetime import datetime, timezone

        path = resolve(self.settings.get("logging", {}).get("path", "data/logs/queries.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record},
                               ensure_ascii=False) + "\n")

    def upload(self, filename: str, data: bytes, step: Any = lambda _: None) -> dict[str, Any]:
        from finquery.ingestion.chunker import text_for_embedding
        from finquery.retrieval.vector_store import ensure_collection, upsert_chunks

        ensure_collection(self.client, self.upload_collection, self.embedder.dimension)

        def embed_and_store(chunks: list[dict[str, Any]]) -> None:
            upsert_chunks(self.client, self.upload_collection, chunks, self.embedder.encode_documents,
                          text_for_embedding)

        doc = self.uploads.add(filename, data, embed_and_store, llm=self.small_llm, step=step)
        self.refresh()
        self._log({"event": "upload", "doc_name": doc["doc_name"], "pages": doc["pages"],
                   "chunks": doc["chunks"]})
        return doc

    def remove(self, doc_name: str) -> None:
        from finquery.retrieval.vector_store import delete_document

        if self.client.collection_exists(self.upload_collection):
            delete_document(self.client, self.upload_collection, doc_name)
        self.uploads.remove(doc_name)
        self.refresh()
