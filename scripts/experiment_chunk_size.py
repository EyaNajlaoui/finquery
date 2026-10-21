"""Compare chunk sizes on the dev set, without touching the Qdrant collection.

For each size the cached pages are chunked again, embedded (vectors cached in
data/processed/experiments/) and searched in memory with numpy.

Usage:
    python scripts/experiment_chunk_size.py                          # 150:30 250:50 400:80
    python scripts/experiment_chunk_size.py --sizes 150:30 250:50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import ROOT, get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.evaluation.retrieval_eval import evaluate
from finquery.ingestion.chunker import text_for_embedding
from finquery.ingestion.pipeline import chunk_document
from finquery.retrieval.embedder import Embedder
from finquery.retrieval.memory_index import NumpyDenseIndex
from finquery.retrieval.reranker import Reranker
from finquery.retrieval.retriever import Retriever


def embed_cached(embedder: Embedder, chunks: list[dict], cache: Path) -> np.ndarray:
    ids_file = cache.with_suffix(".ids.json")
    ids = [c["chunk_id"] + c["content_hash"] for c in chunks]
    if cache.exists() and ids_file.exists() and json.loads(ids_file.read_text()) == ids:
        return np.load(cache)
    print(f"  embedding {len(chunks)} chunks (cached afterwards)...")
    vectors = embedder.encode_documents([text_for_embedding(c) for c in chunks], show_progress=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vectors)
    ids_file.write_text(json.dumps(ids))
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", default=["150:30", "250:50", "400:80"],
                        help="max_words:overlap_words")
    args = parser.parse_args()

    settings = get_settings()
    data = settings["data"]
    processed = resolve(data["processed_dir"])
    catalog = read_jsonl(resolve(data["documents_file"]))
    pages = {d["doc_name"]: read_jsonl(processed / "pages" / f"{d['doc_name']}.jsonl") for d in catalog}
    questions = read_jsonl(resolve(data["eval_dir"]) / "dev.jsonl")

    embedder = Embedder.from_settings(settings)
    reranker = Reranker.from_settings(settings)

    rows = []
    for size in args.sizes:
        max_words, overlap = map(int, size.split(":"))
        chunks = [c for d in catalog for c in chunk_document(pages[d["doc_name"]], d, max_words, overlap)]
        print(f"\n{max_words} words / {overlap} overlap: {len(chunks)} chunks")
        lengths = embedder.count_tokens([text_for_embedding(c) for c in chunks])
        too_long = sum(n > embedder.max_tokens for n in lengths)
        vectors = embed_cached(embedder, chunks, processed / "experiments" / f"w{max_words}_o{overlap}.npy")
        index = NumpyDenseIndex(chunks, vectors)
        retriever = Retriever(chunks, embedder, index.search, reranker,
                              settings["retrieval"]["candidates"], settings["retrieval"].get("rrf_k", 60))
        start = time.perf_counter()
        summary = evaluate(retriever, questions)["summary"]
        print(f"  evaluated in {time.perf_counter() - start:.0f} s")
        rows.append((size, len(chunks), too_long, summary))

    for setting in ("corpus", "document"):
        print(f"\n== {setting} ==  Recall@4 / MRR@10 per mode")
        print(f"{'size':<10}{'chunks':>8}{'>512 tok':>10}" + "".join(f"{m:>22}" for m in
                                                               ("dense", "bm25", "hybrid", "hybrid_rerank")))
        for size, n, too_long, summary in rows:
            cells = []
            for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
                s = summary[f"{setting}/{mode}"]
                cells.append(f"{s['recall@4']:.2f} / {s['mrr@10']:.2f}")
            print(f"{size:<10}{n:>8}{too_long:>10}" + "".join(f"{c:>22}" for c in cells))

    out = ROOT / "results" / "chunk_size_dev.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([{"size": s, "chunks": n, "over_512_tokens": t, "summary": sm}
                               for s, n, t, sm in rows], indent=1), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
