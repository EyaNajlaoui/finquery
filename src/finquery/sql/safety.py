"""First line of defence for LLM-written SQL. The second one is a read-only connection."""
from __future__ import annotations

import re

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|GRANT|TRUNCATE)\b",
    re.IGNORECASE,
)


def clean_sql(text: str) -> str:
    """Remove ```sql fences and a trailing semicolon."""
    text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
    return text.rstrip(";").strip()


def check_sql(sql: str) -> str | None:
    """Return None if the query looks safe, otherwise the reason it is rejected."""
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return "the query must start with SELECT or WITH"
    if ";" in sql:
        return "only one statement is allowed"
    if FORBIDDEN.search(sql):
        return "forbidden keyword (the database is read-only)"
    return None
