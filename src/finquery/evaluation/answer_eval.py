"""Answer evaluation: LLM judge for correctness, code for everything that can be checked
exactly (numbers, citations, whether the evidence was in the context)."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from finquery.evaluation.metrics import contains_number, extract_numbers
from finquery.generation.prompts import JUDGE_SYSTEM, JUDGE_USER

GRADES = ("correct", "partial", "incorrect")
GRADE_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge(llm: Any, question: str, reference: str, candidate: str) -> dict[str, str]:
    response = llm.chat(JUDGE_SYSTEM, JUDGE_USER.format(question=question, reference=reference,
                                                        candidate=candidate), json_mode=True)
    match = re.search(r"\{.*\}", response.text, re.DOTALL)  # tolerate text around the JSON
    try:
        verdict = json.loads(match.group(0) if match else response.text)
        grade = str(verdict.get("grade", "")).lower()
    except (json.JSONDecodeError, AttributeError):
        verdict, grade = {}, ""
    if grade not in GRADES:
        return {"grade": "incorrect", "reason": f"unparsable judge output: {response.text[:200]}"}
    return {"grade": grade, "reason": str(verdict.get("reason", ""))}


def numeric_match(reference: str, candidate: str) -> bool | None:
    """For references that are a single number ('$11588.00', '5.4%'): is that number in the
    candidate (1% tolerance)? None when the reference is not a single number."""
    numbers = extract_numbers(reference)
    if len(numbers) != 1 or len(reference.split()) > 2:
        return None
    return contains_number(candidate, numbers[0])


def code_checks(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    evidence = {(e["doc_name"], e["page"]) for e in question["evidence"]}
    given = {(p["doc_name"], p["page"]) for p in result["passages"]}
    cited = {tuple(c) for c in result["citations"]}
    return {
        "evidence_in_context": bool(evidence & given),
        "has_citation": bool(cited),
        "cites_evidence": bool(evidence & cited),
        "invalid_citations": len(result["invalid_citations"]),
        "numeric_match": numeric_match(question["answer"], result["answer"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """rows: one per question with keys question_type, judge, checks, refusal, seconds, usage."""
    n = len(rows)
    grades = Counter(r["judge"]["grade"] for r in rows)
    numeric = [r["checks"]["numeric_match"] for r in rows if r["checks"]["numeric_match"] is not None]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(GRADE_SCORE[r["judge"]["grade"]])

    # Diagnosis: was the evidence retrieved, and was the answer right?
    matrix = Counter((r["checks"]["evidence_in_context"], r["judge"]["grade"] == "correct") for r in rows)
    return {
        "questions": n,
        "correct": grades["correct"] / n,
        "partial": grades["partial"] / n,
        "incorrect": grades["incorrect"] / n,
        "score": sum(GRADE_SCORE[r["judge"]["grade"]] for r in rows) / n,
        "refusals": sum(r["refusal"] for r in rows) / n,
        "evidence_in_context": sum(r["checks"]["evidence_in_context"] for r in rows) / n,
        "answers_with_citation": sum(r["checks"]["has_citation"] for r in rows) / n,
        "cites_evidence_page": sum(r["checks"]["cites_evidence"] for r in rows) / n,
        "invalid_citations": sum(r["checks"]["invalid_citations"] for r in rows),
        "numeric_match": (sum(numeric) / len(numeric)) if numeric else None,
        "score_by_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
        "diagnosis": {
            "evidence found, answer correct": matrix[(True, True)],
            "evidence found, answer not correct (generation problem)": matrix[(True, False)],
            "evidence missing, answer correct": matrix[(False, True)],
            "evidence missing, answer not correct (retrieval problem)": matrix[(False, False)],
        },
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        "avg_prompt_tokens": round(sum(r["usage"].get("prompt_tokens", 0) for r in rows) / n),
    }
