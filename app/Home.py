"""FinQuery: Streamlit entry point (navigation between the views).

Run locally:   streamlit run app/Home.py
With Docker:   docker compose up --build   then open http://localhost:8501
Online:        Streamlit Community Cloud, main file app/Home.py, GROQ_API_KEY in the app secrets
"""
import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud passes secrets through st.secrets; locally the .env file is used
try:
    for key in ("GROQ_API_KEY", "QDRANT_URL"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
except Exception:  # no secrets.toml: nothing to copy
    pass

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "src"))
sys.path.insert(0, str(APP_DIR))

st.set_page_config(page_title="FinQuery", page_icon=":material/query_stats:", layout="wide")

pages = [
    st.Page(str(APP_DIR / "views" / "chat.py"), title="Ask", icon=":material/forum:", default=True),
    st.Page(str(APP_DIR / "views" / "evaluation.py"), title="Evaluation", icon=":material/monitoring:"),
    st.Page(str(APP_DIR / "views" / "documents.py"), title="Documents", icon=":material/folder_open:"),
]
st.navigation(pages, position="top").run()
