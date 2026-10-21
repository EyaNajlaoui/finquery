"""Metadata filters shared by the dense and the BM25 search.

A filter is a dict such as {"company": "Amazon", "doc_period": [2017, 2019]}:
a single value means "equal", a list means "one of".
"""
from __future__ import annotations

from typing import Any

FILTER_FIELDS = ("doc_name", "company", "doc_type", "doc_period")


def check_filters(filters: dict[str, Any] | None) -> dict[str, list[Any]]:
    """Validate and normalize to {field: [allowed values]}; empty values are ignored."""
    normalized = {}
    for field, value in (filters or {}).items():
        if field not in FILTER_FIELDS:
            raise ValueError(f"Unknown filter field {field!r}, expected one of {FILTER_FIELDS}")
        values = value if isinstance(value, (list, tuple, set)) else [value]
        if values := [v for v in values if v is not None and v != ""]:
            normalized[field] = list(values)
    return normalized


def matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    return all(chunk.get(field) in values for field, values in check_filters(filters).items())
