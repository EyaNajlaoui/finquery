"""Evaluation: the measured results behind every design choice (dev set)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import ui

ui.setup_page()
st.title("Evaluation")
st.caption("FinanceBench dev questions (q = number of questions). "
           "The 30 test questions are kept aside for one final run.")



def system_name(summary: dict, filename: str) -> str:
    """'Agent · gpt-oss-20b · 10 q', from the summary (or the file name for older runs)."""
    parts = filename.removeprefix("generation_dev_").removesuffix(".json").split("_")
    system = summary.get("system") or ("agent" if parts[0] == "agent" else "rag")
    context = summary.get("context") or parts[0]
    kind = "Agent (tools)" if system == "agent" else f"RAG · {context} context"
    model = str(summary.get("model") or "openai/gpt-oss-120b").split("/")[-1]
    return f"{kind} · {model} · {summary['questions']} q"


# --- answer quality --------------------------------------------------------------------------
st.subheader("Answer quality", divider="gray")
rows, summaries = [], {}
for path in sorted((ui.ROOT / "results").glob("generation_dev_*.json")):
    s = ui.load_json(path.name)
    if s and s.get("setting", "corpus") == "corpus":
        name = system_name(s, path.name)
        summaries[name] = s
        rows.append({"System": name, "Score": s["score"], "Correct": s["correct"],
                     "Refusals": s["refusals"], "Cites evidence page": s["cites_evidence_page"],
                     "Tokens / question": s["avg_prompt_tokens"], "Seconds": s["avg_seconds"]})
if rows:
    best = max(rows, key=lambda r: r["Score"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Best score", f"{best['Score']:.0%}", help=f"{best['System']} (correct = 1, partial = 0.5)")
    c2.metric("Correct answers", f"{best['Correct']:.0%}", help=best["System"])
    c3.metric("Cites the evidence page", f"{best['Cites evidence page']:.0%}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        c: st.column_config.ProgressColumn(c, format="percent", min_value=0, max_value=1)
        for c in ("Score", "Correct", "Refusals", "Cites evidence page")} | {
        "Tokens / question": st.column_config.NumberColumn(format="%d"),
        "Seconds": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Compare rows with the same model and number of questions (q): the agent and the RAG "
               "baseline on the same 10 questions with gpt-oss-20b is the like-for-like comparison.")
    diag = summaries[best["System"]]["diagnosis"]
    st.markdown(f"**Where the errors come from** ({best['System']})")
    st.dataframe(pd.DataFrame([{"Case": k, "Questions": v} for k, v in diag.items()]),
                 hide_index=True, width="stretch")
else:
    st.info("Run `python scripts/eval_generation.py` to fill this section.")

# --- retrieval ----------------------------------------------------------------------------------
st.subheader("Retrieval", divider="gray")
retrieval = ui.load_json("retrieval_dev.json")
if retrieval:
    table = pd.DataFrame([{"Setting": run.split("/")[0], "Mode": run.split("/")[1], **values}
                          for run, values in retrieval["summary"].items()])
    st.caption("Is the FinanceBench evidence page among the retrieved chunks? "
               "Corpus = all 18 filings, document = only the question's filing.")
    chart = table.pivot(index="Mode", columns="Setting", values="recall@4").reindex(
        ["dense", "bm25", "hybrid", "hybrid_rerank"])
    st.markdown("**Recall@4 by search mode**")
    st.bar_chart(chart, horizontal=True, stack=False, color=["#6366F1", "#C7C9FA"],
                 x_label="", y_label="", height=260)
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in table.columns if "@" in c})
else:
    st.info("Run `python scripts/eval_retrieval.py` to fill this section.")

chunk = ui.load_json("chunk_size_dev.json")
if chunk:
    st.markdown("**Chunk size** (hybrid + rerank)")
    st.dataframe(pd.DataFrame([{
        "Words / overlap": r["size"].replace(":", " / "), "Chunks": r["chunks"],
        "Over 512 tokens": r["over_512_tokens"],
        "Corpus R@4": r["summary"]["corpus/hybrid_rerank"]["recall@4"],
        "Corpus MRR@10": r["summary"]["corpus/hybrid_rerank"]["mrr@10"],
        "Document R@4": r["summary"]["document/hybrid_rerank"]["recall@4"]} for r in chunk]),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.2f")
                       for c in ("Corpus R@4", "Corpus MRR@10", "Document R@4")})
