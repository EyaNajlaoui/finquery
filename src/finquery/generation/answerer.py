"""Retrieval + generation: question -> grounded answer with citations."""
from __future__ import annotations

import re
from typing import Any

from finquery.generation.context import ContextBuilder, format_passages
from finquery.generation.prompts import ANSWER_SYSTEM, ANSWER_USER

# Square brackets as asked in the prompt, but gpt-oss often writes 【...】: accept both
CITATION = re.compile(r"[\[【]\s*([A-Za-z0-9_.\-]+)\s+p\.\s*(\d+)\s*[\]】]")
REFUSAL = "do not contain enough information"


def parse_citations(text: str) -> list[tuple[str, int]]:
    """[AMAZON_2017_10K p.38] -> ("AMAZON_2017_10K", 38), unique, in order of appearance."""
    seen: dict[tuple[str, int], None] = {}
    for doc, page in CITATION.findall(text):
        seen[(doc, int(page))] = None
    return list(seen)


def check_citations(text: str, passages: list[dict[str, Any]]) -> dict[str, list[tuple[str, int]]]:
    """Citations found in the answer, and those pointing to a page the model never received."""
    given = {(p["doc_name"], p["page"]) for p in passages}
    citations = parse_citations(text)
    return {"citations": citations, "invalid_citations": [c for c in citations if c not in given]}


def is_refusal(text: str) -> bool:
    return REFUSAL in text.lower()


class Answerer:
    def __init__(self, retriever: Any, context: ContextBuilder, llm: Any,
                 k: int = 4, mode: str = "hybrid_rerank") -> None:
        self.retriever, self.context, self.llm = retriever, context, llm
        self.k, self.mode = k, mode

    def answer(self, question: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        results = self.retriever.search(question, k=max(self.k, 10), mode=self.mode, filters=filters)
        passages = self.context.build(results)
        response = self.llm.chat(ANSWER_SYSTEM, ANSWER_USER.format(context=format_passages(passages),
                                                                   question=question))
        return {
            "question": question,
            "answer": response.text,
            **check_citations(response.text, passages),
            "refusal": is_refusal(response.text),
            "passages": passages,
            "retrieved": [(c["doc_name"], c["page"]) for c, _ in results],
            "model": response.model,
            "seconds": round(response.seconds, 2),
            "usage": response.usage,
        }
