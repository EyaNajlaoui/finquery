"""Turn retrieved chunks into the passages given to the LLM.

Modes:
- "chunks": the top k chunks as they are (smallest context)
- "window": each retrieved chunk plus its neighbours on the same page, merged into one
            passage per page. A balance sheet split over 3 chunks is put back together,
            so the LLM sees the column headers (years) with the numbers.
- "page":   the whole page of each retrieved chunk (largest context)

Passages are ordered by the rank of their best chunk, one passage per page,
at most `max_passages`.
"""
from __future__ import annotations

from typing import Any

from finquery.ingestion.chunker import doc_type_label

MODES = ("chunks", "window", "page")
Result = tuple[dict[str, Any], float]


def merge_overlapping(texts: list[str]) -> str:
    """Join consecutive chunks, dropping the lines repeated by the chunk overlap."""
    lines: list[str] = []
    for text in texts:
        new = text.splitlines()
        overlap = next((k for k in range(min(len(lines), len(new)), 0, -1) if lines[-k:] == new[:k]), 0)
        lines += new[overlap:]
    return "\n".join(lines)


class ContextBuilder:
    def __init__(self, chunks: list[dict[str, Any]], pages: dict[tuple[str, int], str] | None = None,
                 mode: str = "window", neighbours: int = 1, max_passages: int = 4) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "page" and pages is None:
            raise ValueError("mode 'page' needs the page texts")
        self.by_page: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
        for c in chunks:
            self.by_page.setdefault((c["doc_name"], c["page"]), {})[c["chunk_index"]] = c
        self.pages = pages or {}
        self.mode, self.neighbours, self.max_passages = mode, neighbours, max_passages

    def build(self, results: list[Result]) -> list[dict[str, Any]]:
        if self.mode == "chunks":
            return [self._passage(c, c["text"]) for c, _ in results[:self.max_passages]]

        hits: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for chunk, _ in results:  # dicts keep insertion order = rank of the best chunk
            key = (chunk["doc_name"], chunk["page"])
            if key in hits or len(hits) < self.max_passages:
                hits.setdefault(key, []).append(chunk)

        passages = []
        for key, page_hits in hits.items():
            first = page_hits[0]
            if self.mode == "page" and key in self.pages:
                passages.append(self._passage(first, self.pages[key]))
                continue
            on_page = self.by_page.get(key, {})
            wanted = sorted({i for c in page_hits
                             for i in range(c["chunk_index"] - self.neighbours,
                                            c["chunk_index"] + self.neighbours + 1) if i in on_page})
            passages.append(self._passage(first, merge_overlapping([on_page[i]["text"] for i in wanted])))
        return passages

    @staticmethod
    def _passage(chunk: dict[str, Any], text: str) -> dict[str, Any]:
        return {"doc_name": chunk["doc_name"], "page": chunk["page"], "company": chunk["company"],
                "doc_type": chunk["doc_type"], "doc_period": chunk["doc_period"], "text": text}


def format_passages(passages: list[dict[str, Any]]) -> str:
    """Passages as the LLM sees them, each starting with the citation tag to use."""
    blocks = []
    for p in passages:
        header = (f"[{p['doc_name']} p.{p['page']}] {p['company']} | {doc_type_label(p['doc_type'])} | "
                  f"FY{p['doc_period']}")
        blocks.append(f"{header}\n{p['text']}")
    return "\n\n---\n\n".join(blocks)
