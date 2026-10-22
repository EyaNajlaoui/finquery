"""Tool-calling agent: the LLM decides what to search and when to calculate.

Why: steps 3-4 showed that most misses are computed metrics (quick ratio, EBITDA margin,
DPO). The question names the metric, the filings only contain its ingredients, often on
two pages. The agent can split the question, search each ingredient in the right filing,
and compute the result with a safe calculator instead of doing arithmetic itself.

Tools:
- search_filings(query, doc_name?)  hybrid search + rerank, passages in "window" mode
- calculate(expression)             safe arithmetic (ast), no Python code can run

The list of available filings is in the system prompt, so the agent can pick the filter
itself (step 3: searching inside the right filing is better than searching everything).
Guardrails: at most `max_steps` model turns, tool errors are returned to the model as
text, citations are checked against every page the tools returned.
"""
from __future__ import annotations

import json
import time
from typing import Any

from finquery.generation.answerer import check_citations, is_refusal
from finquery.generation.context import ContextBuilder, format_passages
from finquery.ingestion.chunker import doc_type_label
from finquery.tools.calculator import CalculatorError, calculate

TOOLS = [
    {"type": "function", "function": {
        "name": "search_filings",
        "description": "Search the company filings. Returns extracts, each starting with its citation tag "
                       "[DOC_NAME p.N]. Use precise financial terms (e.g. 'total current liabilities "
                       "balance sheet'). Search each needed figure separately.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for"},
            "doc_name": {"type": "string", "description": "restrict to one filing, e.g. AMAZON_2017_10K "
                                                          "(see the list of filings). Omit to search all."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression with + - * / ** ( ). Numbers only, "
                       "e.g. '(55900 - 48200) / 48200 * 100'. Use it for every calculation.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
]

AGENT_SYSTEM = """You are FinQuery, a financial analyst assistant with tools.

Available documents ("FY" = calendar year in which the fiscal year ends):
{catalog}

How to work:
1. Identify the filing(s) the question is about and pass doc_name to search_filings.
2. If the question names a metric or ratio (quick ratio, EBITDA margin, DPO...), use your knowledge \
of its definition to list the line items needed, then search each one separately, using the \
statement where it appears (balance sheet, income statement, cash flow statement).
3. Numbers must come ONLY from search results, never from memory. Check the fiscal year and the \
unit (thousands, millions) in the column headers.
4. Do every calculation with the calculate tool.
5. Fiscal year labels differ: a retailer's "fiscal 2022" ending January 2023 is FY2023. Match \
periods by end date and say which period you used.
6. If after searching the information is still missing, say: "The provided documents do not \
contain enough information to answer." and what is missing.

Final answer: one direct sentence first, then the formula and inputs if any. Cite every number \
with square brackets exactly as in the extract headers, e.g. [AMAZON_2017_10K p.38]. Be concise. \
This is analysis, not investment advice."""

FINAL_NUDGE = ("Step limit reached. Do not call tools anymore: give your final answer now with what "
               "you found, or say what is missing.")


def catalog_text(catalog: list[dict[str, Any]]) -> str:
    lines = []
    for d in catalog:
        parts = [d.get("company") or d.get("title") or "",
                 doc_type_label(d["doc_type"]) if d.get("doc_type") not in (None, "", "other") else "",
                 f"FY{d['doc_period']}" if d.get("doc_period") else "fiscal year unknown"]
        lines.append(f"- {d['doc_name']}: " + ", ".join(p for p in parts if p))
    return "\n".join(lines)


class Agent:
    def __init__(self, retriever: Any, context: ContextBuilder, llm: Any, catalog: list[dict[str, Any]],
                 max_steps: int = 6, search_k: int = 10, mode: str = "hybrid_rerank",
                 max_chars_per_search: int = 4000, max_total_chars: int = 16000) -> None:
        self.retriever, self.context, self.llm = retriever, context, llm
        # Size guardrails: every tool output is sent again at each turn, so the prompt grows.
        # Groq's free tier also caps tokens per minute (8,000 for gpt-oss-20b) per request.
        self.max_chars_per_search, self.max_total_chars = max_chars_per_search, max_total_chars
        self.used_chars = 0
        self.catalog = catalog
        self.scope: list[str] = [d["doc_name"] for d in catalog]  # documents the agent may search
        self.max_steps, self.search_k, self.mode = max_steps, search_k, mode

    # --- tools -------------------------------------------------------------------------
    def _search(self, args: dict[str, Any], seen: dict[tuple[str, int], dict[str, Any]]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "Error: 'query' is required."
        doc_name = args.get("doc_name") or None
        if doc_name and doc_name not in self.scope:
            return f"Error: unknown doc_name {doc_name!r}. Use one from the list of filings."
        results = self.retriever.search(query, k=self.search_k, mode=self.mode,
                                        filters={"doc_name": doc_name or self.scope})
        passages = self.context.build(results)
        if not passages:
            return "No results."
        if self.used_chars >= self.max_total_chars:
            return ("Search budget used up: answer now with the extracts you already have, "
                    "or say what is missing.")
        for p in passages:
            seen.setdefault((p["doc_name"], p["page"]), p)
        # Long pages are shortened around the start (tables and titles come first)
        cap = max(400, self.max_chars_per_search // len(passages))
        shown = [{**p, "text": p["text"] if len(p["text"]) <= cap else p["text"][:cap] + "\n[...]"}
                 for p in passages]
        output = format_passages(shown)
        self.used_chars += len(output)
        return output

    @staticmethod
    def _calculate(args: dict[str, Any]) -> str:
        try:
            return str(calculate(str(args.get("expression", ""))))
        except CalculatorError as error:
            return f"Error: {error}"

    def run_tool(self, name: str, arguments: str, seen: dict) -> str:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "Error: arguments must be valid JSON."
        if name == "search_filings":
            return self._search(args, seen)
        if name == "calculate":
            return self._calculate(args)
        return f"Error: unknown tool {name!r}."

    # --- loop --------------------------------------------------------------------------
    def answer(self, question: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Same output keys as Answerer.answer, plus the trace of tool calls.
        `filters` (doc_name: one name or a list, company) limits the filings the agent can search,
        as when the user selects documents in the app."""
        selected = (filters or {}).get("doc_name")
        selected = [selected] if isinstance(selected, str) else list(selected or [])
        company = (filters or {}).get("company")
        docs = [d for d in self.catalog if (not selected or d["doc_name"] in selected)
                and (not company or d.get("company") == company)] or self.catalog
        self.scope = [d["doc_name"] for d in docs]  # searches stay inside the user's selection
        system = AGENT_SYSTEM.format(catalog=catalog_text(docs))
        messages: list[dict[str, Any]] = [{"role": "system", "content": system},
                                          {"role": "user", "content": question}]
        seen: dict[tuple[str, int], dict[str, Any]] = {}
        self.used_chars = 0
        trace, usage, start, text = [], {"prompt_tokens": 0, "completion_tokens": 0}, time.perf_counter(), ""

        for step in range(1, self.max_steps + 2):
            if step == self.max_steps + 1:  # guardrail: one last turn to conclude
                messages.append({"role": "user", "content": FINAL_NUDGE})
            turn = self.llm.complete(messages, TOOLS)
            for key in usage:
                usage[key] += turn.usage.get(key, 0)
            if not turn.tool_calls:
                text = turn.content
                break
            if step == self.max_steps + 1:
                text = "The agent could not finish within the step limit."
                break
            messages.append(turn.as_message())
            for call in turn.tool_calls:
                output = self.run_tool(call.name, call.arguments, seen)
                trace.append({"step": step, "tool": call.name, "arguments": call.arguments,
                              "output_preview": output[:300]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        passages = list(seen.values())
        return {
            "question": question,
            "answer": text,
            **check_citations(text, passages),
            "refusal": is_refusal(text),
            "passages": passages,
            "retrieved": [(p["doc_name"], p["page"]) for p in passages],
            "trace": trace,
            "steps": len({t["step"] for t in trace}) + 1,
            "model": self.llm.model,
            "seconds": round(time.perf_counter() - start, 2),
            "usage": usage,
        }
