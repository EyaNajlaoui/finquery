"""PDF -> clean text, one entry per page.

pdfplumber's `extract_text` keeps each table row on one line
("Total current assets 29,505 28,120"), which is what retrieval needs.
Extracting tables separately with `find_tables` was tested and rejected:
it duplicates the numbers, splits "$" and the amount into different cells,
and loses the column headers (years), which stay in the page text.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"[ \t]+")

EDGE_LINES = 2        # header/footer zone: first and last lines of each page
MIN_SHARE = 0.3       # a line is boilerplate if it is in that zone on >= 30% of pages...
MIN_PAGES = 3         # ...and on at least 3 pages


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Raw text of each page (index 0 = page 1)."""
    import pdfplumber  # imported here so the rest of the package works without it

    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _lines(text: str) -> list[str]:
    return [_SPACES.sub(" ", line).strip() for line in text.splitlines() if line.strip()]


def _signature(line: str) -> str:
    """'Page 38 of 120' and 'Page 39 of 120' share the signature 'page # of #'."""
    return _DIGITS.sub("#", line.lower())


def find_boilerplate(pages: list[str]) -> set[str]:
    """Signatures of lines repeated in the header/footer zone (page numbers,
    'Table of Contents', company name...)."""
    counts: Counter[str] = Counter()
    for text in pages:
        lines = _lines(text)
        edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({_signature(line) for line in edge})
    threshold = max(MIN_PAGES, MIN_SHARE * len(pages))
    return {sig for sig, n in counts.items() if n >= threshold}


def clean_pages(pages: list[str]) -> list[str]:
    """Normalize spaces and drop boilerplate lines, only in the header/footer zone,
    so that a repeated line inside a table ('Net sales') is never removed."""
    boilerplate = find_boilerplate(pages)
    cleaned = []
    for text in pages:
        lines = _lines(text)
        n = len(lines)
        if n <= 2 * EDGE_LINES:  # very short page: everything would be "edge", keep it all
            cleaned.append("\n".join(lines))
            continue
        kept = [line for i, line in enumerate(lines)
                if not ((i < EDGE_LINES or i >= n - EDGE_LINES) and _signature(line) in boilerplate)]
        cleaned.append("\n".join(kept))
    return cleaned


def parse_pdf(pdf_path: str | Path) -> list[str]:
    """Clean text of each page of a PDF (index 0 = page 1)."""
    return clean_pages(extract_pages(pdf_path))
