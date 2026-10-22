"""Shared helpers for the Streamlit views: cached resources and formatting."""
from __future__ import annotations

import html
import json
import os
import re
import string
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from finquery.config import ROOT, get_settings, resolve
from finquery.data.financebench import read_jsonl
from finquery.ingestion.chunker import doc_type_label

CITATION = re.compile(r"[\[【]\s*([A-Za-z0-9_.\-]+)\s+p\.\s*(\d+)\s*[\]】]")

# Colour themes: the sidebar is dark, the work area light, one accent colour for actions and citations.
PALETTES = {
    "emerald": {"side": "#0B1730", "side2": "#14254A", "side_text": "#C7D2E3", "accent": "#10B981",
                "accent_dark": "#047857", "accent_soft": "#E6F7F0", "user": "#0B1730"},
    "indigo": {"side": "#1A1745", "side2": "#2A2665", "side_text": "#D4D2F2", "accent": "#6366F1",
               "accent_dark": "#4338CA", "accent_soft": "#EEF0FF", "user": "#1A1745"},
}
PALETTE = PALETTES.get(os.getenv("FINQUERY_PALETTE", "indigo"), PALETTES["indigo"])

CSS = "<style>" + string.Template((Path(__file__).parent / "style.css").read_text(encoding="utf-8")) \
    .substitute(PALETTE) + "</style>"


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


def owner() -> str:
    """A random id for this browser session: uploads are private to the visitor who added them.
    A new tab or a page reload starts a new session (and an empty document list)."""
    return st.session_state.setdefault("owner", uuid.uuid4().hex)


def catalog() -> list[dict[str, Any]]:
    """This visitor's uploads first, then the sample library. Read from disk (cheap), so it
    works before the models are loaded."""
    settings = get_settings()
    uploads = resolve(settings["uploads"]["dir"]) / "catalog.jsonl"
    library = resolve(settings["data"]["documents_file"])
    mine = [d for d in (read_jsonl(uploads) if uploads.exists() else []) if d.get("owner") == owner()]
    return ([{**d, "source": "upload"} for d in mine]
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


def badge_label(doc_name: str) -> str:
    """Short name for citation badges: the company (or title) only, the full name is in the tooltip."""
    for doc in catalog():
        if doc["doc_name"] == doc_name:
            return doc.get("company") or doc.get("title") or doc_name
    return doc_name.replace("_", " ")


def render_answer(text: str, given: set[tuple[str, int]]) -> str:
    """Answer as HTML-safe markdown: '$' escaped (Streamlit reads $...$ as LaTeX) and
    citations shown as badges; red badge for a page the model never received.
    Render with st.markdown(..., unsafe_allow_html=True)."""
    text = html.escape(text, quote=False).replace("$", "\\$")  # the model's text is never raw HTML

    def badge(match: re.Match) -> str:
        doc, page = match.group(1), int(match.group(2))
        css = "fq-cite" if (doc, page) in given else "fq-cite bad"
        return f'<span class="{css}" title="{short_label(doc)}">{badge_label(doc)} · p.{page}</span>'

    return CITATION.sub(badge, text)


def settings_ready() -> list[str]:
    """What is missing before the chat can work, as instructions."""
    get_settings()  # loads .env
    problems = []
    if os.getenv("FINQUERY_SECRETS_ERROR"):
        problems.append("The app secrets could not be read (check the TOML format, the value needs "
                        f'quotes: `GROQ_API_KEY = "gsk_..."`): {os.environ["FINQUERY_SECRETS_ERROR"]}')
    if not os.getenv("GROQ_API_KEY", "").strip().startswith("gsk_"):
        problems.append("Add your Groq key: locally in `.env` (`GROQ_API_KEY=gsk_...`), on Streamlit Cloud "
                        'in **Settings → Secrets** (`GROQ_API_KEY = "gsk_..."`), then reboot the app.')
    return problems



def snippet(page_text: str, answer: str, max_chars: int = 230) -> str:
    """The lines of a source page that contain numbers used in the answer (else its first lines),
    so the sources panel shows *why* the page matters."""
    numbers = {n for n in re.findall(r"\d[\d,.]*\d|\d", answer) if len(n.replace(",", "")) >= 3}
    lines = [line for line in page_text.splitlines() if line.strip()]
    hits = [line for line in lines if any(n in line for n in numbers)]
    text = " … ".join(hits[:3]) if hits else " ".join(lines[:4])
    text = text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + " …"
    return html.escape(text)
