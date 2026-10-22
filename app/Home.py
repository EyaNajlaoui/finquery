"""FinQuery: Streamlit entry point (navigation between the views).

Run locally:   streamlit run app/Home.py
With Docker:   docker compose up --build   then open http://localhost:8501
Online:        Streamlit Community Cloud, main file app/Home.py, GROQ_API_KEY in the app secrets
"""
import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud passes secrets through st.secrets; locally the .env file is used.
# Keys are accepted at the top level or inside a [section] of the secrets.
try:
    secrets = {k: v for k, v in st.secrets.items()}
    for value in list(secrets.values()):
        if hasattr(value, "items"):
            secrets.update({k: v for k, v in value.items()})
    for key in ("GROQ_API_KEY", "QDRANT_URL"):
        if secrets.get(key) and not os.getenv(key):
            os.environ[key] = str(secrets[key]).strip()
except Exception as error:
    # no secrets at all (local run): .env is used; badly formatted secrets: say so in the app
    if not str(error).startswith("No secrets found"):
        os.environ["FINQUERY_SECRETS_ERROR"] = str(error).split(": ")[-1]

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
