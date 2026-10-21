"""Parse the PDFs of data/raw and build data/processed/chunks.jsonl.

Usage:
    python scripts/ingest.py              # parsed pages are cached, re-runs are fast
    python scripts/ingest.py --reparse    # force PDF parsing again
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery.config import get_settings, resolve
from finquery.data.financebench import read_jsonl, write_jsonl
from finquery.ingestion.pipeline import chunk_document, load_or_parse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reparse", action="store_true", help="ignore the parsed-pages cache")
    args = parser.parse_args()

    settings = get_settings()
    data, chunking = settings["data"], settings["chunking"]
    raw_dir, processed = resolve(data["raw_dir"]), resolve(data["processed_dir"])
    catalog = read_jsonl(resolve(data["documents_file"]))

    all_chunks, total_pages, start = [], 0, time.perf_counter()
    print(f"{'document':<42}{'pages':>6}{'chunks':>8}{'seconds':>9}")
    for doc in catalog:
        pdf = raw_dir / f"{doc['doc_name']}.pdf"
        if not pdf.exists():
            sys.exit(f"Missing {pdf.name}: run python scripts/download_data.py first")
        t = time.perf_counter()
        pages = load_or_parse(pdf, processed / "pages", reparse=args.reparse)
        chunks = chunk_document(pages, doc, chunking["max_words"], chunking["overlap_words"])
        all_chunks += chunks
        total_pages += len(pages)
        print(f"{doc['doc_name']:<42}{len(pages):>6}{len(chunks):>8}{time.perf_counter() - t:>9.1f}")

    write_jsonl(all_chunks, processed / "chunks.jsonl")
    words = [len(c["text"].split()) for c in all_chunks]
    print(f"\n{len(catalog)} documents, {total_pages} pages, {len(all_chunks)} chunks "
          f"(avg {sum(words) / len(words):.0f} words, max {max(words)}) "
          f"in {time.perf_counter() - start:.0f} s -> {processed / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
