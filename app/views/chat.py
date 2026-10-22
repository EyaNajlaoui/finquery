"""Ask: upload any financial document (or use the sample library) and ask questions."""
from __future__ import annotations

import streamlit as st
import ui

from finquery.config import get_settings
from finquery.ingestion.uploads import SUPPORTED, UploadError

UPLOAD_EXAMPLES = [
    "Summarize the key financial results of this document.",
    "What were total revenue and net income, and how did they change versus the prior year?",
    "What is the gross margin for the latest fiscal year? Show the calculation.",
    "What are the main risks or uncertainties mentioned?",
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


def handle_upload(file) -> None:
    """Index an uploaded file once, then select it for the next questions."""
    if file is None or st.session_state.get("last_upload") == file.file_id:
        return
    st.session_state.last_upload = file.file_id
    if file.size > settings["uploads"]["max_mb"] * 1e6:
        st.error(f"This file is larger than {settings['uploads']['max_mb']} MB.")
        return
    system = ui.get_system()
    with st.status(f"Adding **{file.name}**...", expanded=True) as status:
        try:
            doc = system.upload(file.name, file.getvalue(), step=status.write)
        except UploadError as error:
            status.update(label=str(error), state="error")
            return
        status.update(label=f"Ready: {ui.doc_label({**doc, 'source': 'upload'})} ({doc['pages']} pages)",
                      state="complete", expanded=False)
    st.session_state.pending_scope = [doc["doc_name"]]  # applied before the widget is drawn
    st.rerun()


# --- sidebar ---------------------------------------------------------------------------------
docs = ui.catalog()
by_name = {d["doc_name"]: d for d in docs}
# By default, search the user's own documents (the sample library only if there are none)
st.session_state.setdefault("scope", [d["doc_name"] for d in docs if d["source"] == "upload"])
if "pending_scope" in st.session_state:
    st.session_state.scope = st.session_state.pop("pending_scope")
st.session_state.scope = [n for n in st.session_state.scope if n in by_name]

with st.sidebar:
    st.markdown("### Your documents")
    sidebar_file = st.file_uploader("Add a financial document", type=FILE_TYPES,
                                    help="Annual or quarterly report, earnings release, 10-K... "
                                         "PDF with selectable text, or a text file.")
    st.multiselect("Search in", list(by_name), key="scope", format_func=lambda n: ui.doc_label(by_name[n]),
                   placeholder="All documents",
                   help="Leave empty to search every document. 📄 = uploaded by you.")
    st.markdown("### Answer")
    mode = st.segmented_control("Mode", ["Agent", "Fast"], default="Agent", key="mode",
                                help="Agent: plans several searches and uses a calculator (slower, "
                                     "better for ratios). Fast: one search, one answer.") or "Agent"
    show_sources = st.toggle("Show sources", value=True)
    st.divider()
    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Answers can be wrong: check the cited pages. Not investment advice.")
handle_upload(sidebar_file)

# --- header ----------------------------------------------------------------------------------
st.markdown(
    '<div class="fq-hero"><h1>FinQuery</h1>'
    "<p>Upload a financial report and ask questions. Every number comes with its source page.</p></div>",
    unsafe_allow_html=True)
uploaded = [d for d in docs if d["source"] == "upload"]
library = [d for d in docs if d["source"] == "library"]
chips = [f"{len(uploaded)} uploaded" if uploaded else "", f"{len(library)} sample filings" if library else "",
         "Hybrid search + reranking", "Cited answers"]
st.markdown("".join(f'<span class="fq-chip">{c}</span>' for c in chips if c), unsafe_allow_html=True)

problems = ui.settings_ready()
if problems:
    st.error("FinQuery is not ready yet:\n\n" + "\n".join(f"- {p}" for p in problems))
    st.stop()


def show_result(result: dict) -> None:
    given = {(p["doc_name"], p["page"]) for p in result["passages"]}
    if result["refusal"]:
        st.warning("The selected documents do not seem to contain this information.",
                   icon=":material/search_off:")
    st.markdown(ui.render_answer(result["answer"], given), unsafe_allow_html=True)

    tokens = sum(result["usage"].values()) if result["usage"] else 0
    meta = [f"{result['seconds']:.1f} s", f"{tokens:,} tokens", f"{len(result['passages'])} pages read"]
    if result.get("trace") is not None:
        meta.append(f"{sum(t['tool'] == 'search_filings' for t in result['trace'])} searches")
    st.markdown(f'<span class="fq-meta">{" · ".join(meta)}</span>', unsafe_allow_html=True)

    if result["invalid_citations"]:
        st.error("Some citations point to pages the model never received (shown in red).",
                 icon=":material/error:")
    if result.get("trace"):
        with st.expander(f"Reasoning steps ({len(result['trace'])})", icon=":material/account_tree:"):
            for t in result["trace"]:
                icon = ":material/search:" if t["tool"] == "search_filings" else ":material/calculate:"
                st.markdown(f"{icon} **Step {t['step']}** · `{t['tool']}`")
                st.code(t["arguments"], language="json")
    if show_sources and result["passages"]:
        cited = {tuple(c) for c in result["citations"]}
        with st.expander(f"Sources ({len(result['passages'])} pages)", icon=":material/description:"):
            for p in result["passages"]:
                status = "cited" if (p["doc_name"], p["page"]) in cited else "read, not cited"
                st.markdown(f"**{ui.short_label(p['doc_name'])} · page {p['page']}** "
                            f'<span class="fq-meta">· {status}</span>', unsafe_allow_html=True)
                with st.container(height=180, border=True):  # scrollable: pages can be long
                    st.text(p["text"])


# --- conversation ----------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user"
                         else ":material/query_stats:"):
        if message["role"] == "user":
            st.markdown(message["content"].replace("$", "\\$"))
        else:
            show_result(message["result"])

question = st.chat_input("Ask about your documents...") or st.session_state.pop("pending", None)
if not st.session_state.messages and not question:
    if not uploaded:
        handle_upload(st.file_uploader("Upload a financial document to start", type=FILE_TYPES,
                                       key="main_upload"))
        if library:
            st.caption(f"Or try the sample library: {len(library)} SEC filings from FinanceBench.")
    in_scope = [by_name[n] for n in st.session_state.scope] or uploaded
    examples = UPLOAD_EXAMPLES if any(d["source"] == "upload" for d in in_scope) else LIBRARY_EXAMPLES
    st.markdown("##### Try one of these")
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
            st.session_state.pending = example  # asked on the next run, once the buttons are gone
            st.rerun()

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question.replace("$", "\\$"))
    with st.chat_message("assistant", avatar=":material/query_stats:"):
        system = ui.get_system()
        label = "Planning searches and calculations..." if mode == "Agent" else "Searching the documents..."
        with st.spinner(label):
            try:
                result = system.ask(question, agent=mode == "Agent", doc_names=st.session_state.scope)
            except Exception as error:  # show the problem instead of a stack trace
                st.error(f"Something went wrong: {error}")
                st.stop()
        show_result(result)
    st.session_state.messages.append({"role": "assistant", "result": result})
