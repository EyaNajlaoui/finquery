"""Page text -> chunks with metadata.

Design choices:
- Chunks never cross a page boundary, so every chunk cites one exact page.
- Text is split on line boundaries: a table row is never cut in half.
- Each chunk gets a context header ("Amazon | 10-K | FY2017 | page 38 |
  CONSOLIDATED STATEMENTS OF OPERATIONS"). A chunk that only contains
  "Total current assets 29,505 28,120" is meaningless without it.
- The full page text is kept separately (pages.jsonl): retrieval works on small
  chunks, the LLM can then read the whole page (small-to-big).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

DOC_TYPE_LABELS = {"10k": "10-K", "10q": "10-Q", "8k": "8-K", "earnings": "Earnings release"}

_LETTERS = re.compile(r"[A-Za-z]")


def doc_type_label(doc_type: str) -> str:
    return DOC_TYPE_LABELS.get(str(doc_type).lower(), str(doc_type))


def page_title(text: str, max_lines: int = 6, max_titles: int = 2) -> str:
    """Lines written in capitals near the top of the page, e.g.
    'COSTCO WHOLESALE CORPORATION - CONSOLIDATED BALANCE SHEETS'.
    Two lines are kept because the first one is often only the company name."""
    titles = []
    for line in text.splitlines()[:max_lines]:
        letters = _LETTERS.findall(line)
        if len(letters) >= 8 and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            titles.append(line.strip())
    return " - ".join(titles[:max_titles])


def split_lines(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Group lines into pieces of at most max_words words. The next piece starts
    with the last lines of the previous one (about overlap_words words).
    A single line longer than max_words is cut into word windows."""
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    lines: list[list[str]] = []
    for line in text.splitlines():
        words = line.split()
        while len(words) > max_words:
            lines.append(words[:max_words])
            words = words[max_words - overlap_words:]
        if words:
            lines.append(words)

    pieces, current = [], []
    for words in lines:
        if current and sum(map(len, current)) + len(words) > max_words:
            pieces.append(current)
            tail, size = [], 0
            for previous in reversed(current):
                if size + len(previous) > overlap_words:
                    break
                tail.insert(0, previous)
                size += len(previous)
            current = tail if size + len(words) <= max_words else []
        current.append(words)
    if current:
        pieces.append(current)
    return ["\n".join(" ".join(words) for words in piece) for piece in pieces]


def make_context(doc: dict[str, Any], page: int, title: str) -> str:
    parts = [doc["company"], doc_type_label(doc["doc_type"]), f"FY{doc['doc_period']}", f"page {page}"]
    return " | ".join(parts + ([title] if title else []))


def chunk_page(text: str, page: int, doc: dict[str, Any],
               max_words: int, overlap_words: int) -> list[dict[str, Any]]:
    """Chunks of one page. `doc` holds doc_name, company, doc_type, doc_period."""
    title = page_title(text)
    context = make_context(doc, page, title)
    chunks = []
    for i, piece in enumerate(split_lines(text, max_words, overlap_words)):
        chunks.append({
            "chunk_id": f"{doc['doc_name']}:p{page}:c{i}",
            "doc_name": doc["doc_name"],
            "company": doc["company"],
            "doc_type": doc["doc_type"],
            "doc_period": doc["doc_period"],
            "page": page,
            "chunk_index": i,
            "context": context,
            "text": piece,
            # changes when the text or the context changes: used for incremental re-indexing
            "content_hash": hashlib.sha1(f"{context}\n{piece}".encode()).hexdigest()[:16],
        })
    return chunks


def text_for_embedding(chunk: dict[str, Any]) -> str:
    """What the embedding model and BM25 see: context header + chunk text."""
    return f"{chunk['context']}\n{chunk['text']}"
