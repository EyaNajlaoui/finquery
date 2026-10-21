"""FinanceBench: download the questions and PDFs, normalize them, split dev/test.

FinanceBench (Patronus AI) is licensed CC BY-NC 4.0. Only the selected documents
listed in config/settings.yaml are downloaded.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

QUESTION_FIELDS = ["financebench_id", "company", "doc_name", "doc_type", "doc_period",
                   "question_type", "question_reasoning", "question", "answer", "justification"]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_documents(questions: list[dict], documents: list[dict]) -> list[dict]:
    """Add document metadata (doc_type, doc_period...) to each question (fallback source)."""
    by_name = {d["doc_name"]: d for d in documents}
    return [{**by_name.get(q["doc_name"], {}), **q} for q in questions]


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep the useful fields. FinanceBench pages are 0-based; we store 1-based pages,
    the numbering used everywhere else in FinQuery."""
    row = {k: raw.get(k) for k in QUESTION_FIELDS}
    evidence = raw.get("evidence") or []
    row["evidence"] = [{"text": e.get("evidence_text", ""),
                        "doc_name": e.get("doc_name") or e.get("evidence_doc_name") or raw["doc_name"],
                        "page": int(e["evidence_page_num"]) + 1}
                       for e in evidence]
    row["evidence_pages"] = sorted({e["page"] for e in row["evidence"]})
    return row


def select(questions: list[dict], documents: list[str]) -> list[dict]:
    wanted = set(documents)
    return [q for q in questions if q["doc_name"] in wanted]


def stratified_split(questions: list[dict], dev_size: int, seed: int,
                     key: str = "question_type") -> tuple[list[dict], list[dict]]:
    """Split into dev/test, keeping the proportion of each question type."""
    if not 0 < dev_size < len(questions):
        raise ValueError("dev_size must be between 1 and len(questions) - 1")
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["financebench_id"]):
        groups[q[key]].append(q)

    # Largest-remainder allocation of the dev quota across groups
    ratio = dev_size / len(questions)
    quotas = {g: int(len(items) * ratio) for g, items in groups.items()}
    remainders = sorted(groups, key=lambda g: len(groups[g]) * ratio - quotas[g], reverse=True)
    for g in remainders[: dev_size - sum(quotas.values())]:
        quotas[g] += 1

    dev, test = [], []
    for g in sorted(groups):
        items = groups[g][:]
        rng.shuffle(items)
        dev += items[: quotas[g]]
        test += items[quotas[g]:]
    return dev, test
