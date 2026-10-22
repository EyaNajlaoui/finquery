"""Documents: the user's uploads, the sample library, and the data pipeline."""
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st
import ui

from finquery.config import get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.ingestion.chunker import doc_type_label

ui.setup_page()
st.title("Documents")
settings = get_settings()
all_docs = ui.catalog()
docs = [d for d in all_docs if d["source"] == "library"]
uploads = [d for d in all_docs if d["source"] == "upload"]

st.subheader("Your documents", divider="gray")
if not uploads:
    st.info("No document uploaded yet: add one from the **Ask** page.", icon=":material/upload_file:")
for d in uploads:
    c1, c2, c3 = st.columns([6, 2, 1], vertical_alignment="center")
    c1.markdown(f"**{ui.doc_label(d)}**  \n<span class='fq-meta'>{d['filename']}</span>",
                unsafe_allow_html=True)
    c2.markdown(f"<span class='fq-meta'>{d['pages']} pages · {d['chunks']} chunks</span>",
                unsafe_allow_html=True)
    if c3.button("", icon=":material/delete:", key=f"rm_{d['doc_name']}", help="Remove this document"):
        ui.get_system().remove(d["doc_name"])
        st.rerun()

st.subheader("Sample library", divider="gray")
st.caption("18 SEC filings from FinanceBench, used to measure FinQuery (see Evaluation).")
raw_dir = resolve(settings["data"]["raw_dir"])
processed = resolve(settings["data"]["processed_dir"])


@st.cache_data(show_spinner=False)
def chunk_counts() -> Counter:
    path = processed / "chunks.jsonl"
    return Counter(c["doc_name"] for c in read_jsonl(path)) if path.exists() else Counter()


@st.cache_data(show_spinner=False)
def page_count(doc_name: str) -> int:
    path = processed / "pages" / f"{doc_name}.jsonl"
    return sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0


chunks = chunk_counts()
vectors = 0
try:
    client = ui.get_client()
    name = settings["vector_store"]["collection"]
    vectors = client.count(name).count if client.collection_exists(name) else 0
except Exception as error:  # the page must still load if Qdrant is down
    st.error(f"Qdrant: {error}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filings", len(docs))
c2.metric("Pages", f"{sum(page_count(d['doc_name']) for d in docs):,}")
c3.metric("Chunks", f"{sum(chunks.values()):,}")
c4.metric("Vectors in Qdrant", f"{vectors:,}")

table = pd.DataFrame([{
    "Company": d["company"], "Type": doc_type_label(d["doc_type"]), "Fiscal year": str(d["doc_period"]),
    "Pages": page_count(d["doc_name"]), "Chunks": chunks.get(d["doc_name"], 0),
    "PDF": (raw_dir / f"{d['doc_name']}.pdf").exists(), "File": d["doc_name"]} for d in docs])
if docs:
    st.dataframe(table.sort_values(["Company", "Fiscal year"]), hide_index=True, width="stretch",
                 height=36 * (len(table) + 1) + 3,
                 column_config={"PDF": st.column_config.CheckboxColumn("PDF downloaded")})
else:
    st.info("Run `python scripts/download_data.py` to download the filings.")

with st.expander("Pipeline", icon=":material/settings:"):
    st.markdown("""
1. `python scripts/download_data.py` — 18 FinanceBench filings and the dev/test questions
2. `python scripts/ingest.py` — PDF text, cleaning, chunks with company / year / page metadata
3. `python scripts/build_index.py` — embeddings in Qdrant (incremental)
4. `python scripts/eval_retrieval.py`, `python scripts/eval_generation.py` — measured on the dev set
""")
    sections = ("chunking", "embedding", "reranker", "retrieval", "generation", "agent", "llm")
    st.json({k: settings[k] for k in sections if k in settings}, expanded=False)
st.caption("Source: FinanceBench (Patronus AI), CC BY-NC 4.0.")
