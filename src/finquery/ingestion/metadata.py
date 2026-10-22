"""Company, document type and fiscal year of an uploaded document.

Step 3 showed that the context header (company, year) is what lets the search find the
right filing among many. For an uploaded file we do not know these fields, so a small LLM
reads the first pages and fills them; without an LLM, simple rules give a best guess.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

DOC_TYPES = ("10k", "10q", "8k", "earnings", "annual report", "other")

METADATA_SYSTEM = """You read the first pages of a financial document and return JSON only:
{"company": "<short company name or empty>", "doc_type": "<one of: 10k, 10q, 8k, earnings, \
annual report, other>", "doc_period": <fiscal year as an integer (the calendar year in which the \
fiscal year ends) or null>, "title": "<short title of the document>"}
Use only what is written in the text. Use empty values when unsure."""

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def guess_metadata(first_pages: str, filename: str) -> dict[str, Any]:
    """Rules only: the most frequent year near the top, the file name as title."""
    years = Counter(int(y) for y in _YEAR.findall(first_pages[:6000]))
    title = re.sub(r"[_\-]+", " ", filename.rsplit(".", 1)[0]).strip()
    text = first_pages.lower()
    doc_type = ("10k" if "form 10-k" in text else "10q" if "form 10-q" in text
                else "8k" if "form 8-k" in text else "other")
    return {"company": "", "doc_type": doc_type,
            "doc_period": years.most_common(1)[0][0] if years else None, "title": title}


def detect_metadata(first_pages: str, filename: str, llm: Any = None) -> dict[str, Any]:
    guess = guess_metadata(first_pages, filename)
    if llm is None:
        return guess
    try:
        response = llm.chat(METADATA_SYSTEM, f"File name: {filename}\n\n{first_pages[:6000]}", json_mode=True)
        found = json.loads(response.text)
    except Exception:  # metadata is a nice-to-have: never block an upload for it
        return guess
    period = found.get("doc_period")
    doc_type = str(found.get("doc_type") or "").lower()
    return {
        "company": str(found.get("company") or "").strip()[:80],
        "doc_type": doc_type if doc_type in DOC_TYPES else guess["doc_type"],
        "doc_period": int(period) if isinstance(period, (int, float)) or str(period).isdigit()
        else guess["doc_period"],
        "title": str(found.get("title") or guess["title"]).strip()[:120],
    }
