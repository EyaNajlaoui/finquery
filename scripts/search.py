"""Try the dense and BM25 searches side by side on one question.

Usage:
    python scripts/search.py "What was Costco's total assets in FY2021?"
    python scripts/search.py "days payable outstanding" --company Amazon --year 2017 -k 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.retrieval.bm25 import BM25Index
from finquery.retrieval.embedder import Embedder
from finquery.retrieval.vector_store import dense_search, get_client


def show(title: str, results: list) -> None:
    print(f"\n=== {title} ===")
    for rank, (chunk, score) in enumerate(results, start=1):
        preview = " ".join(chunk["text"].split())[:150]
        print(f"{rank}. [{chunk['doc_name']} p.{chunk['page']}] score={score:.3f}\n   {preview}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5, help="results per method")
    parser.add_argument("--company")
    parser.add_argument("--year", type=int, help="fiscal year (doc_period)")
    parser.add_argument("--doc-type", help="10k, 10q, 8k or Earnings")
    args = parser.parse_args()

    filters = {"company": args.company, "doc_period": args.year, "doc_type": args.doc_type}
    settings = get_settings()

    chunks = read_jsonl(resolve(settings["data"]["processed_dir"]) / "chunks.jsonl")
    show("BM25", BM25Index(chunks).search(args.question, args.k, filters))

    embedder = Embedder.from_settings(settings)
    client = get_client(settings)
    vector = embedder.encode_query(args.question)
    show("Dense (Qdrant)", dense_search(client, settings["vector_store"]["collection"], vector,
                                        args.k, filters))
    client.close()


if __name__ == "__main__":
    main()
