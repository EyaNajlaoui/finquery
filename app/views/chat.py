"""Ask: upload a financial document (or use the sample library) and chat with it.

Layout: dark sidebar (documents, mode) | conversation | sources of the last answer.
"""
from __future__ import annotations

import streamlit as st
import ui

from finquery.config import get_settings
from finquery.ingestion.uploads import SUPPORTED, UploadError

UPLOAD_EXAMPLES = [
    "Summarize the key financial results of this document.",
    "What were revenue and net income, and how did they change versus the prior year?",
    "What is the gross margin for the latest fiscal year? Show the calculation.",
    "What are the main risks mentioned?",
]
LIBRARY_EXAMPLES = [
    "What were Costco's total assets at the end of FY2021?",
    "Has AMCOR's quick ratio improved or declined between FY2023 and FY2022?",
    "Has Boeing reported any materially important ongoing legal battles from FY2022?",
    "What is Amazon's year-over-year change in revenue from FY2016 to FY2017?",
]
FILE_TYPES = [s.strip(".") for s in SUPPORTED]

ui.setup_page()
settings = get_settings()
st.session_state.setdefault("messages", [])
docs = ui.catalog()
by_name = {d["doc_name"]: d for d in docs}
uploaded = [d for d in docs if d["source"] == "upload"]
library = [d for d in docs if d["source"] == "library"]
# Which documents are searched: one checkbox per document. New uploads start checked,
# sample filings unchecked; with nothing checked, every document is searched.
for d in uploaded:
    st.session_state.setdefault(f"sel_{d['doc_name']}", True)
for d in library:
    st.session_state.setdefault(f"sel_{d['doc_name']}", False)


def set_all(names: list[str], value: bool) -> None:
    for name in names:
        st.session_state[f"sel_{name}"] = value


def handle_upload(file) -> None:
    """Index an uploaded file once, then check it."""
    if file is None or st.session_state.get("last_upload") == file.file_id:
        return
    st.session_state.last_upload = file.file_id
    if file.size > settings["uploads"]["max_mb"] * 1e6:
        st.error(f"This file is larger than {settings['uploads']['max_mb']} MB.")
        return
    with st.status(f"Adding **{file.name}**...", expanded=True) as status:
        try:
            doc = ui.get_system().upload(file.name, file.getvalue(), step=status.write,
                                          owner=ui.owner())
        except UploadError as error:
            status.update(label=str(error), state="error")
            return
        status.update(label=f"Ready: {ui.doc_label({**doc, 'source': 'upload'})}", state="complete")
    st.session_state[f"sel_{doc['doc_name']}"] = True
    st.session_state.uploader_n = st.session_state.get("uploader_n", 0) + 1  # empties the upload widget
    st.rerun()


def document_list(group: list[dict], key: str) -> None:
    names = [d["doc_name"] for d in group]
    left, right = st.columns(2)
    left.button("Select all", key=f"all_{key}", width="stretch", on_click=set_all, args=(names, True))
    right.button("Clear", key=f"none_{key}", width="stretch", on_click=set_all, args=(names, False))
    for d in group:
        st.checkbox(ui.doc_label(d).removeprefix("📄 "), key=f"sel_{d['doc_name']}")


# --- sidebar ---------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="fq-logo"><div class="mark">F</div><div class="name">FinQuery</div></div>',
                unsafe_allow_html=True)
    sidebar_file = st.file_uploader("Add a document", type=FILE_TYPES, label_visibility="collapsed",
                                    key=f"side_upload_{st.session_state.get('uploader_n', 0)}",
                                    help="Annual or quarterly report, earnings release, 10-K... "
                                         "PDF with selectable text, or a text file.")
    checked_mine = sum(st.session_state[f"sel_{d['doc_name']}"] for d in uploaded)
    st.markdown(f'<div class="fq-side-label">MY DOCUMENTS · {checked_mine}/{len(uploaded)} SELECTED</div>',
                unsafe_allow_html=True)
    if uploaded:
        document_list(uploaded, "mine")
    else:
        st.caption("Upload a PDF to start.")
    st.caption(f":material/lock: Private to you: other visitors cannot see your documents. They are "
               f"deleted after {settings['uploads'].get('max_age_hours', 2)} hours, and a page reload "
               "starts a new session.")
    if library:
        checked_lib = sum(st.session_state[f"sel_{d['doc_name']}"] for d in library)
        with st.expander(f"Sample library · {checked_lib}/{len(library)} selected"):
            document_list(library, "lib")

    st.markdown('<div class="fq-side-label">ANSWER MODE</div>', unsafe_allow_html=True)
    mode = st.segmented_control("Mode", ["Agent", "Fast"], default="Agent", key="mode",
                                label_visibility="collapsed",
                                help="Agent: plans several searches and uses a calculator "
                                     "(better for ratios). Fast: one search, one answer.") or "Agent"
    show_sources = st.toggle("Show sources panel", value=True)
    chunks = sum(d.get("chunks", 0) for d in uploaded)
    st.markdown(f'<div class="fq-status"><span class="fq-dot"></span>Index up to date'
                f'{f" · {chunks:,} chunks" if chunks else ""}</div>', unsafe_allow_html=True)
st.session_state.scope = [d["doc_name"] for d in docs if st.session_state.get(f"sel_{d['doc_name']}")]
handle_upload(sidebar_file)

problems = ui.settings_ready()
if problems:
    st.error("FinQuery is not ready yet:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()

# --- top bar ---------------------------------------------------------------------------------
scope_docs = [by_name[n] for n in st.session_state.scope]
if len(scope_docs) == 1:
    d = scope_docs[0]
    title = ui.doc_label(d).removeprefix("📄 ")
    sub = f"{d.get('filename', d['doc_name'])} · {d.get('pages', '?')} pages"
elif scope_docs:
    title = f"{len(scope_docs)} documents"
    sub = " · ".join(ui.short_label(d["doc_name"]) for d in scope_docs[:3])
else:
    title = "All documents"
    sub = f"No document selected: searching all {len(uploaded)} uploaded and {len(library)} sample filings"
left, right = st.columns([2.3, 1], gap="large") if show_sources else (st.container(), None)

with left:
    bar, button = st.columns([5, 1.4], vertical_alignment="center")
    bar.markdown(f'<div class="fq-topbar"><div><div class="title">{title}'
                 f'<span class="fq-pill">{mode}</span></div><div class="sub">{sub}</div></div></div>',
                 unsafe_allow_html=True)
    if button.button("New chat", icon=":material/add_comment:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    chat = st.container(height=560, border=False)
    with chat:
        if not st.session_state.messages:
            st.markdown(
                '<div class="fq-hero"><h2>Ask anything about your financial documents</h2>'
                "<p>Hybrid search finds the right pages, an agent computes the ratios, "
                "and every number links to its source page.</p>"
                '<div class="kpis"><div class="kpi"><b>Hybrid search</b><span>embeddings + BM25 + reranking'
                '</span></div><div class="kpi"><b>Agent</b><span>multi-step search + calculator</span></div>'
                '<div class="kpi"><b>Cited pages</b><span>every number linked to its source</span></div>'
                "</div></div>",
                unsafe_allow_html=True)
            if not uploaded:
                handle_upload(st.file_uploader("Upload a financial document to start", type=FILE_TYPES,
                                               key="main_upload"))
            examples = UPLOAD_EXAMPLES if any(d["source"] == "upload" for d in scope_docs or uploaded) \
                else LIBRARY_EXAMPLES
            st.markdown('<div class="fq-section">TRY ONE OF THESE</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, example in enumerate(examples):
                if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
                    st.session_state.pending = example
                    st.rerun()
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"].replace("$", "\\$"))
                continue
            result = message["result"]
            with st.chat_message("assistant", avatar=":material/query_stats:"):
                if result["refusal"]:
                    st.warning("The selected documents do not seem to contain this information.",
                               icon=":material/search_off:")
                given = {(p["doc_name"], p["page"]) for p in result["passages"]}
                st.markdown(ui.render_answer(result["answer"], given), unsafe_allow_html=True)
                tokens = sum(result["usage"].values()) if result["usage"] else 0
                st.markdown(f'<span class="fq-meta">{result["seconds"]:.1f} s · {tokens:,} tokens · '
                            f'{len(result["passages"])} pages read</span>', unsafe_allow_html=True)
                if result["invalid_citations"]:
                    st.error("Some citations point to pages the model never received (in red).")
        if st.session_state.get("thinking"):
            with st.chat_message("assistant", avatar=":material/query_stats:"):
                label = "Planning searches and calculations..." if mode == "Agent" else "Searching..."
                with st.spinner(label):
                    question = st.session_state.pop("thinking")
                    try:
                        result = ui.get_system().ask(question, agent=mode == "Agent",
                                                     doc_names=st.session_state.scope, owner=ui.owner())
                        st.session_state.messages.append({"role": "assistant", "result": result})
                    except Exception as error:  # show the problem instead of a stack trace
                        st.session_state.messages.append({"role": "assistant", "result": {
                            "answer": f"Something went wrong: {error}", "passages": [], "citations": [],
                            "invalid_citations": [], "refusal": False, "usage": {}, "seconds": 0.0}})
                st.rerun()

    question = st.chat_input("Ask a question about your documents...") \
        or st.session_state.pop("pending", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.thinking = question
        st.rerun()
    st.markdown('<div class="fq-meta" style="text-align:center">Answers come only from the selected '
                "documents and can be wrong: check the cited pages. Not investment advice.</div>",
                unsafe_allow_html=True)

# --- sources panel ---------------------------------------------------------------------------
if right is not None:
    with right:
        st.markdown('<div class="fq-section">SOURCES OF THE LAST ANSWER</div>', unsafe_allow_html=True)
        last = next((m["result"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                    None)
        if not last or not last["passages"]:
            st.markdown('<div class="fq-note">Sources appear here after your first question: every page the '
                        "model read, with the lines it used.</div>", unsafe_allow_html=True)
        else:
            cited = {tuple(c) for c in last["citations"]}
            ordered = sorted(last["passages"], key=lambda p: (p["doc_name"], p["page"]) not in cited)
            for p in ordered:
                used = (p["doc_name"], p["page"]) in cited
                st.markdown(
                    f'<div class="fq-source"><div class="head"><span class="doc">'
                    f'{ui.short_label(p["doc_name"])}</span><span class="page">Page {p["page"]}</span></div>'
                    f'<div class="quote">{ui.snippet(p["text"], last["answer"])}</div>'
                    f'<div class="status{"" if used else " read"}">'
                    f'{"● Cited in the answer" if used else "○ Read, not cited"}</div></div>',
                    unsafe_allow_html=True)
            if last.get("trace"):
                st.markdown('<div class="fq-section" style="margin-top:1rem">REASONING STEPS</div>',
                            unsafe_allow_html=True)
                for t in last["trace"]:
                    icon = "🔎" if t["tool"] == "search_filings" else "🧮"
                    st.markdown(f'<div class="fq-step">{icon} <b>{t["tool"]}</b> '
                                f'<code>{ui.html.escape(t["arguments"][:90])}</code></div>',
                                unsafe_allow_html=True)
            st.markdown('<div class="fq-note">Every number is followed by a badge with its page. A red '
                        "badge means a page the model never received.</div>", unsafe_allow_html=True)
