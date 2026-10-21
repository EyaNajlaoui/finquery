"""Load config/settings.yaml, with environment variable overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config" / "settings.yaml"


def _load_dotenv() -> None:
    """Read .env if python-dotenv is installed (optional in CI)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def get_settings(path: str | Path | None = None) -> dict[str, Any]:
    _load_dotenv()
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    # Environment overrides
    if os.getenv("QDRANT_URL"):
        settings["vector_store"]["url"] = os.environ["QDRANT_URL"]
    if os.getenv("FINQUERY_LLM_MODEL"):
        settings["llm"]["model"] = os.environ["FINQUERY_LLM_MODEL"]
    return settings


def resolve(relative: str) -> Path:
    """Paths in settings.yaml are relative to the project root."""
    return ROOT / relative
