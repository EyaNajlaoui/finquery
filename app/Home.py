"""FinQuery: Streamlit entry point.

Run locally:   streamlit run app/Home.py
With Docker:   docker compose up --build   then open http://localhost:8501
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finquery import __version__
from finquery.config import get_settings, resolve
from finquery.data.financebench import read_jsonl

st.set_page_config(page_title="FinQuery", page_icon="📊", layout="wide")
settings = get_settings()

st.title("FinQuery")
st.caption(f"Question answering over financial filings with RAG · v{__version__}")
st.warning("FinQuery is an analysis tool, not investment advice.")

st.subheader("Project status")
st.info("Steps 0-2 are in place: data, ingestion and indexing. The chat interface arrives in step 6.")

col_data, col_eval, col_store = st.columns(3)

raw_dir = resolve(settings["data"]["raw_dir"])
pdfs = sorted(raw_dir.glob("*.pdf"))
expected = len(settings["data"]["documents"])
col_data.metric("Documents downloaded", f"{len(pdfs)} / {expected}")

eval_dir = resolve(settings["data"]["eval_dir"])
counts = {name: len(read_jsonl(eval_dir / f"{name}.jsonl")) if (eval_dir / f"{name}.jsonl").exists() else 0
          for name in ("dev", "test")}
col_eval.metric("Evaluation questions", f"{counts['dev']} dev · {counts['test']} test")

store = settings["vector_store"]


@st.cache_resource  # one client per process: local Qdrant locks its folder
def get_qdrant():
    from qdrant_client import QdrantClient
    if store["url"]:
        return QdrantClient(url=store["url"])
    return QdrantClient(path=str(resolve(store["local_path"])))


vectors = 0
try:
    client = get_qdrant()
    vectors = client.count(store["collection"]).count if client.collection_exists(store["collection"]) else 0
    col_store.metric("Vectors in " + ("Qdrant server" if store["url"] else "Qdrant local"), f"{vectors:,}")
except Exception as error:  # the app must still start if Qdrant is down
    col_store.metric("Vector store", "unavailable")
    st.error(f"Qdrant: {error}")

if len(pdfs) < expected or not counts["dev"]:
    st.markdown("Run `python scripts/download_data.py` to download the data.")
elif not vectors:
    st.markdown("Run `python scripts/ingest.py` then `python scripts/build_index.py` to build the index.")

with st.expander("Configuration"):
    sections = ("chunking", "embedding", "reranker", "retrieval", "llm", "vector_store")
    st.json({k: settings[k] for k in sections})
