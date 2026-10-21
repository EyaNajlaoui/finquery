"""Retrieval evaluation: are the FinanceBench evidence pages among the retrieved chunks?

Two settings:
- "corpus":   search all 18 documents (the system must also find the right filing)
- "document": search only the question's filing (as when the user picks it in the UI)
"""
from __future__ import annotations

from typing import Any

from finquery.evaluation.metrics import recall_at_k, reciprocal_rank

SETTINGS = ("corpus", "document")


def filters_for(question: dict[str, Any], setting: str) -> dict[str, Any] | None:
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    return {"doc_name": question["doc_name"]} if setting == "document" else None


def hit_keys(results: list[tuple[dict[str, Any], float]]) -> list[tuple[str, int]]:
    """(document, page) of each retrieved chunk, in rank order."""
    return [(chunk["doc_name"], chunk["page"]) for chunk, _ in results]


def evaluate(retriever: Any, questions: list[dict[str, Any]], ks: tuple[int, ...] = (1, 4, 10),
             settings: tuple[str, ...] = SETTINGS) -> dict[str, Any]:
    """Per-question first-hit ranks and mean Recall@k / MRR per (setting, mode)."""
    k_max = max(ks)
    per_question = []
    for q in questions:
        relevant = sorted({(e["doc_name"], e["page"]) for e in q["evidence"]})
        row = {"financebench_id": q["financebench_id"], "question_type": q["question_type"],
               "relevant": relevant, "runs": {}}
        for setting in settings:
            all_modes = retriever.search_all_modes(q["question"], k_max, filters_for(q, setting))
            for mode, results in all_modes.items():
                keys = hit_keys(results)
                first = next((i for i, key in enumerate(keys, start=1) if key in relevant), None)
                row["runs"][f"{setting}/{mode}"] = {
                    "first_hit": first, "retrieved": keys,
                    **{f"recall@{k}": recall_at_k(keys, relevant, k) for k in ks},
                    f"mrr@{k_max}": reciprocal_rank(keys, relevant, k_max)}
        per_question.append(row)

    summary = {}
    for run in per_question[0]["runs"] if per_question else []:
        metrics = [m for m in per_question[0]["runs"][run] if m not in ("first_hit", "retrieved")]
        summary[run] = {m: sum(r["runs"][run][m] for r in per_question) / len(per_question)
                        for m in metrics}
    return {"summary": summary, "questions": per_question}


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    if not summary:
        return "(no results)"
    metrics = list(next(iter(summary.values())))
    lines = [f"{'setting/mode':<26}" + "".join(f"{m:>12}" for m in metrics)]
    lines += [f"{run:<26}" + "".join(f"{values[m]:>12.2f}" for m in metrics)
              for run, values in summary.items()]
    return "\n".join(lines)
