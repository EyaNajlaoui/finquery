"""Embed data/processed/chunks.jsonl and store the vectors in Qdrant.

Incremental: only new or changed chunks are embedded, deleted chunks are removed.

Usage:
    python scripts/build_index.py              # incremental
    python scripts/build_index.py --rebuild    # delete the collection and start again
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.ingestion.chunker import text_for_embedding
from finquery.retrieval.embedder import Embedder
from finquery.retrieval.vector_store import ensure_collection, get_client, sync_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="delete the collection first")
    args = parser.parse_args()

    settings = get_settings()
    chunks_path = resolve(settings["data"]["processed_dir"]) / "chunks.jsonl"
    if not chunks_path.exists():
        sys.exit("No chunks found: run python scripts/ingest.py first")
    chunks = read_jsonl(chunks_path)

    print(f"Loading {settings['embedding']['model']}... (downloaded once, then cached)")
    embedder = Embedder.from_settings(settings)

    # Chunks longer than the model limit are silently cut: check it instead of assuming
    lengths = embedder.count_tokens([text_for_embedding(c) for c in chunks])
    too_long = sum(n > embedder.max_tokens for n in lengths)
    print(f"Tokens per chunk: avg {sum(lengths) / len(lengths):.0f}, max {max(lengths)} "
          f"| over the {embedder.max_tokens}-token limit: {too_long} chunks")

    client = get_client(settings)
    name = settings["vector_store"]["collection"]
    if args.rebuild and client.collection_exists(name):
        client.delete_collection(name)
        print(f"Collection {name!r} deleted")
    ensure_collection(client, name, embedder.dimension)

    start = time.perf_counter()

    def progress(done: int, total: int) -> None:
        print(f"\r  embedded {done}/{total}", end="", flush=True)

    counts = sync_chunks(client, name, chunks, embedder.encode_documents, text_for_embedding,
                         progress=progress)
    print(f"\n{counts} in {time.perf_counter() - start:.0f} s")
    print(f"Collection {name!r}: {client.count(name).count} vectors")
    client.close()


if __name__ == "__main__":
    main()
