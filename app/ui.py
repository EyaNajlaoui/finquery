"""Shared helpers for the Streamlit views: cached resources and formatting."""
from __future__ import annotations

import html
import json
import os
import re
from typing import Any

import streamlit as st

from finquery.config import ROOT, get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.ingestion.chunker import doc_type_label

CITATION = re.compile(r"[\[【]\s*([A-Za-z0-9_.\-]+)\s+p\.\s*(\d+)\s*[\]】]")

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1100px;}
.fq-hero h1 {font-size: 2.1rem; margin-bottom: 0;}
.fq-hero p {color: #475569; margin-top: .2rem;}
.fq-chip {display: inline-block; padding: .15rem .6rem; margin: 0 .35rem .35rem 0; border-radius: 999px;
          background: #E6F2F0; color: #0F766E; font-size: .8rem; font-weight: 600;}
.fq-cite {display: inline-block; padding: 0 .45rem; border-radius: 6px; background: #E6F2F0;
          color: #0F766E; font-size: .78rem; font-weight: 600; white-space: nowrap;}
.fq-cite.bad {background: #FDECEC; color: #B42318;}
.fq-meta {color: #64748B; font-size: .8rem;}
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_client() -> Any:
    """One Qdrant client per process (local Qdrant locks its folder)."""
    from finquery.retrieval.vector_store import get_client as make_client

    return make_client(get_settings())


@st.cache_resource(show_spinner="Loading the search models (first time only)...")
def get_system() -> Any:
    from finquery.generation.pipeline import FinQuerySystem

    return FinQuerySystem(get_settings(), client=get_client())


def catalog() -> list[dict[str, Any]]:
    """Uploaded documents first, then the sample library. Read from disk (cheap), so it
    works before the models are loaded."""
    settings = get_settings()
    uploads = resolve(settings["uploads"]["dir"]) / "catalog.jsonl"
    library = resolve(settings["data"]["documents_file"])
    return ([{**d, "source": "upload"} for d in (read_jsonl(uploads) if uploads.exists() else [])]
            + [{**d, "source": "library"} for d in (read_jsonl(library) if library.exists() else [])])


def doc_label(doc: dict[str, Any]) -> str:
    parts = [doc.get("company") or doc.get("title") or doc["doc_name"],
             doc_type_label(doc["doc_type"]) if doc.get("doc_type") and doc["doc_type"] != "other" else "",
             f"FY{doc['doc_period']}" if doc.get("doc_period") else ""]
    label = " · ".join(p for p in parts if p)
    return f"📄 {label}" if doc.get("source") == "upload" else label


def load_json(name: str) -> Any:
    path = ROOT / "results" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def short_label(doc_name: str) -> str:
    """AMAZON_2017_10K -> 'Amazon · 10-K · FY2017' (falls back to the raw name)."""
    for doc in catalog():
        if doc["doc_name"] == doc_name:
            return doc_label(doc).removeprefix("📄 ")
    return doc_name.replace("_", " ")


def render_answer(text: str, given: set[tuple[str, int]]) -> str:
    """Answer as HTML-safe markdown: '$' escaped (Streamlit reads $...$ as LaTeX) and
    citations shown as badges; red badge for a page the model never received.
    Render with st.markdown(..., unsafe_allow_html=True)."""
    text = html.escape(text, quote=False).replace("$", "\\$")  # the model's text is never raw HTML

    def badge(match: re.Match) -> str:
        doc, page = match.group(1), int(match.group(2))
        css = "fq-cite" if (doc, page) in given else "fq-cite bad"
        return f'<span class="{css}" title="{doc}">{short_label(doc)} · p.{page}</span>'

    return CITATION.sub(badge, text)


def settings_ready() -> list[str]:
    """What is missing before the chat can work, as instructions."""
    get_settings()  # loads .env
    problems = []
    if not os.getenv("GROQ_API_KEY", "").startswith("gsk_"):
        problems.append("Add your Groq key to `.env`: `GROQ_API_KEY=gsk_...`")
    return problems

