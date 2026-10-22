"""Documents uploaded by the user in the app.

They go through the same pipeline as the benchmark filings (parse -> clean -> chunk with
context header -> embed), but are stored apart:
- files, pages, chunks and catalog in data/uploads/
- vectors in their own Qdrant collection
so the FinanceBench evaluation is never mixed with user documents.

Each upload belongs to one visitor (`owner`, a random id per browser session): a visitor only
sees, searches and deletes their own documents, and uploads are deleted after a few hours.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from finquery.data.financebench import read_jsonl, write_jsonl
from finquery.ingestion.chunker import chunk_page
from finquery.ingestion.metadata import detect_metadata
from finquery.ingestion.parser import clean_pages, parse_pdf

SUPPORTED = (".pdf", ".txt", ".md")
WORDS_PER_TEXT_PAGE = 500


class UploadError(ValueError):
    pass


def text_pages(text: str) -> list[str]:
    """Plain text has no pages: cut it every ~500 words, on line boundaries."""
    pages, current, size = [], [], 0
    for line in text.splitlines():
        current.append(line)
        size += len(line.split())
        if size >= WORDS_PER_TEXT_PAGE:
            pages.append("\n".join(current))
            current, size = [], 0
    if any(line.strip() for line in current):
        pages.append("\n".join(current))
    return pages


def make_doc_name(filename: str, data: bytes, owner: str = "") -> str:
    """'Annual Report 2024.pdf' -> 'UP_ANNUAL_REPORT_2024_3f2a1c9b' (stable for the same file and
    owner: two visitors uploading the same file get two separate documents)."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_").upper()[:40] or "DOCUMENT"
    return f"UP_{stem}_{hashlib.sha1(owner.encode() + data).hexdigest()[:8]}"


class UploadStore:
    def __init__(self, root: Path, max_words: int, overlap_words: int) -> None:
        self.root = root
        self.files, self.pages_dir = root / "files", root / "pages"
        self.catalog_path, self.chunks_path = root / "catalog.jsonl", root / "chunks.jsonl"
        self.max_words, self.overlap_words = max_words, overlap_words

    # --- reading -------------------------------------------------------------------------
    def catalog(self, owner: str | None = None) -> list[dict[str, Any]]:
        """All uploads (owner=None) or only one visitor's."""
        docs = read_jsonl(self.catalog_path) if self.catalog_path.exists() else []
        return docs if owner is None else [d for d in docs if d.get("owner") == owner]

    def expired(self, max_age_hours: float, now: float | None = None) -> list[str]:
        """Uploads older than max_age_hours (documents without a date count as expired)."""
        limit = (now or time.time()) - max_age_hours * 3600
        return [d["doc_name"] for d in self.catalog() if d.get("added_at", 0) < limit]

    def chunks(self) -> list[dict[str, Any]]:
        return read_jsonl(self.chunks_path) if self.chunks_path.exists() else []

    def pages(self) -> dict[tuple[str, int], str]:
        found = {}
        for path in self.pages_dir.glob("*.jsonl") if self.pages_dir.exists() else []:
            for p in read_jsonl(path):
                found[p["doc_name"], p["page"]] = p["text"]
        return found

    # --- writing -------------------------------------------------------------------------
    def parse(self, filename: str, data: bytes, doc_name: str) -> list[str]:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED:
            raise UploadError(f"Unsupported file type {suffix}: use {', '.join(SUPPORTED)}")
        self.files.mkdir(parents=True, exist_ok=True)
        path = self.files / f"{doc_name}{suffix}"  # unique name: two "report.pdf" never collide
        path.write_bytes(data)
        if suffix == ".pdf":
            pages = parse_pdf(path)
        else:
            pages = clean_pages(text_pages(data.decode("utf-8", errors="replace")))
        if sum(len(p.split()) for p in pages) < 30:
            raise UploadError("No text found in this file. Scanned PDFs (images) need OCR, "
                              "which FinQuery does not do yet.")
        return pages

    def add(self, filename: str, data: bytes, embed_and_store: Callable[[list[dict]], None],
            llm: Any = None, step: Callable[[str], None] = lambda _: None, owner: str = "") -> dict[str, Any]:
        """Parse, describe, chunk and index one file for one visitor. Returns its catalog entry.
        `embed_and_store(chunks)` writes the vectors (the caller owns the models)."""
        doc_name = make_doc_name(filename, data, owner)
        existing = next((d for d in self.catalog() if d["doc_name"] == doc_name), None)
        if existing:
            return existing

        step("Reading the pages")
        pages = self.parse(filename, data, doc_name)
        step("Detecting company, document type and fiscal year")
        doc = {"doc_name": doc_name, "filename": Path(filename).name,
               "stored_as": f"{doc_name}{Path(filename).suffix.lower()}", "pages": len(pages),
               "owner": owner, "added_at": int(time.time()),
               **detect_metadata("\n".join(pages[:3]), filename, llm)}
        chunks = [c for i, text in enumerate(pages, start=1) if text.strip()
                  for c in chunk_page(text, i, doc, self.max_words, self.overlap_words)]
        doc["chunks"] = len(chunks)

        step(f"Embedding {len(chunks)} chunks")
        embed_and_store(chunks)
        write_jsonl([{"doc_name": doc_name, "page": i, "text": t} for i, t in enumerate(pages, start=1)],
                    self.pages_dir / f"{doc_name}.jsonl")
        write_jsonl(self.chunks() + chunks, self.chunks_path)
        write_jsonl(self.catalog() + [doc], self.catalog_path)
        return doc

    def remove(self, doc_name: str) -> None:
        doc = next((d for d in self.catalog() if d["doc_name"] == doc_name), None)
        write_jsonl([d for d in self.catalog() if d["doc_name"] != doc_name], self.catalog_path)
        write_jsonl([c for c in self.chunks() if c["doc_name"] != doc_name], self.chunks_path)
        (self.pages_dir / f"{doc_name}.jsonl").unlink(missing_ok=True)
        if doc and doc.get("stored_as"):
            (self.files / doc["stored_as"]).unlink(missing_ok=True)

