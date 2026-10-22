import json

import pytest

from finquery.generation.llm import LLMResponse
from finquery.ingestion.chunker import make_context
from finquery.ingestion.metadata import detect_metadata, guess_metadata
from finquery.ingestion.uploads import UploadError, UploadStore, make_doc_name, text_pages

REPORT = "\n".join(["ACME CORPORATION", "ANNUAL REPORT 2024", "Fiscal year ended December 31, 2024"]
                   + [f"Revenue line {i}: total revenue grew to 1,200 million in 2024" for i in range(120)])


def test_doc_name_is_stable_and_safe():
    assert make_doc_name("Annual Report (2024).pdf", b"x") == make_doc_name("Annual Report (2024).pdf", b"x")
    assert make_doc_name("Annual Report (2024).pdf", b"x").startswith("UP_ANNUAL_REPORT_2024_")
    assert make_doc_name("a.pdf", b"x") != make_doc_name("a.pdf", b"y")


def test_text_pages_split():
    pages = text_pages(REPORT)
    assert len(pages) >= 2 and all(p.strip() for p in pages)


def test_guess_metadata_without_llm():
    meta = guess_metadata(REPORT, "acme_annual_report.pdf")
    assert meta["doc_period"] == 2024 and meta["title"] == "acme annual report"


class JsonLLM:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, system, user, json_mode=False):
        return LLMResponse(self.payload, "small", 0.1, {})


def test_detect_metadata_with_llm_and_bad_output():
    good = json.dumps({"company": "Acme", "doc_type": "annual report", "doc_period": 2024,
                       "title": "Acme AR"})
    assert detect_metadata(REPORT, "f.pdf", JsonLLM(good))["company"] == "Acme"
    assert detect_metadata(REPORT, "f.pdf", JsonLLM("not json"))["doc_period"] == 2024  # falls back


def test_context_header_with_missing_fields():
    doc = {"doc_name": "UP_X_1", "company": "", "doc_type": "other", "doc_period": None, "title": "Acme AR"}
    assert make_context(doc, 3, "") == "Acme AR | page 3"


def test_upload_store_add_and_remove(tmp_path):
    store = UploadStore(tmp_path, max_words=150, overlap_words=30)
    stored = []
    doc = store.add("acme.txt", REPORT.encode(), stored.extend)
    assert doc["doc_period"] == 2024 and doc["chunks"] == len(stored) > 0
    assert store.catalog() == [doc] and len(store.chunks()) == doc["chunks"]
    assert all(c["doc_name"] == doc["doc_name"] for c in stored)
    assert store.add("acme.txt", REPORT.encode(), stored.extend) == doc  # same file: not added twice
    store.remove(doc["doc_name"])
    assert store.catalog() == [] and store.chunks() == [] and store.pages() == {}


def test_upload_rejects_empty_and_unsupported(tmp_path):
    store = UploadStore(tmp_path, 150, 30)
    with pytest.raises(UploadError):
        store.add("empty.txt", b"   ", lambda chunks: None)
    with pytest.raises(UploadError):
        store.add("sheet.xlsx", b"data", lambda chunks: None)


def test_uploads_are_private_to_their_owner(tmp_path):
    store = UploadStore(tmp_path, 150, 30)
    alice = store.add("acme.txt", REPORT.encode(), lambda _: None, owner="alice")
    bob = store.add("acme.txt", REPORT.encode(), lambda _: None, owner="bob")
    assert alice["doc_name"] != bob["doc_name"]  # same file, two private copies
    assert store.catalog("alice") == [alice] and store.catalog("bob") == [bob]
    assert store.catalog("carol") == [] and len(store.catalog()) == 2


def test_old_uploads_expire(tmp_path):
    store = UploadStore(tmp_path, 150, 30)
    doc = store.add("acme.txt", REPORT.encode(), lambda _: None, owner="alice")
    assert store.expired(2) == []
    assert store.expired(2, now=doc["added_at"] + 3 * 3600) == [doc["doc_name"]]
