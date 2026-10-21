"""Ingestion of a set of documents: parse (with a per-document cache), then chunk."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from finquery.data.financebench import read_jsonl, write_jsonl
from finquery.ingestion.chunker import chunk_page
from finquery.ingestion.parser import parse_pdf


def load_or_parse(pdf_path: Path, cache_dir: Path, reparse: bool = False) -> list[dict[str, Any]]:
    """Pages of a PDF as [{"doc_name", "page", "text"}]. Parsing is the slow part,
    so the result is cached in cache_dir/<doc_name>.jsonl."""
    cache = cache_dir / f"{pdf_path.stem}.jsonl"
    if cache.exists() and not reparse:
        return read_jsonl(cache)
    pages = [{"doc_name": pdf_path.stem, "page": i, "text": text}
             for i, text in enumerate(parse_pdf(pdf_path), start=1)]
    write_jsonl(pages, cache)
    return pages


def chunk_document(pages: list[dict[str, Any]], doc: dict[str, Any],
                   max_words: int, overlap_words: int) -> list[dict[str, Any]]:
    return [chunk for p in pages if p["text"].strip()
            for chunk in chunk_page(p["text"], p["page"], doc, max_words, overlap_words)]
