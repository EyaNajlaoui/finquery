"""Search tests with a tiny hand-made embedding, so no model is downloaded.
Qdrant runs in memory (":memory:")."""
import numpy as np
import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from qdrant_client import QdrantClient  # noqa: E402

from finquery.ingestion.chunker import chunk_page, text_for_embedding  # noqa: E402
from finquery.retrieval.bm25 import BM25Index, tokenize  # noqa: E402
from finquery.retrieval.filters import check_filters, matches  # noqa: E402
from finquery.retrieval.vector_store import (  # noqa: E402
    dense_search,
    ensure_collection,
    stored_hashes,
    sync_chunks,
)

VOCAB = ["assets", "revenue", "debt", "employees"]


def embed(texts):
    """Bag-of-words over 4 words: enough to check the plumbing."""
    vectors = np.array([[t.lower().count(w) + 0.01 for w in VOCAB] for t in texts], dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_chunks():
    docs = [
        ({"doc_name": "COSTCO_2021_10K", "company": "Costco", "doc_type": "10k", "doc_period": 2021},
         ["Total assets 59,268", "Revenue 195,929", "Employees 288,000"]),
        ({"doc_name": "AMAZON_2017_10K", "company": "Amazon", "doc_type": "10k", "doc_period": 2017},
         ["Total assets 131,310", "Long-term debt 24,743"]),
    ]
    return [c for doc, pages in docs for page, text in enumerate(pages, start=1)
            for c in chunk_page(text, page, doc, max_words=150, overlap_words=30)]


@pytest.fixture
def client():
    client = QdrantClient(":memory:")
    ensure_collection(client, "test", dimension=len(VOCAB))
    yield client
    client.close()


def test_filters():
    assert check_filters({"company": "Amazon", "doc_period": [2017, 2019], "doc_type": None}) == {
        "company": ["Amazon"], "doc_period": [2017, 2019]}
    assert matches({"company": "Amazon", "doc_period": 2017}, {"doc_period": [2017, 2019]})
    with pytest.raises(ValueError):
        check_filters({"ceo": "x"})


def test_bm25_tokenize_keeps_numbers_and_drops_stopwords():
    assert tokenize("What was the total of 59,268?") == ["total", "59268"]


def test_bm25_search_and_filter():
    index = BM25Index(make_chunks())
    top, _ = index.search("total assets", limit=2)[0]
    assert "assets" in top["text"].lower()
    results = index.search("total assets", limit=5, filters={"company": "Amazon"})
    assert results and all(c["company"] == "Amazon" for c, _ in results)


def test_sync_is_incremental(client):
    chunks = make_chunks()
    assert sync_chunks(client, "test", chunks, embed, text_for_embedding) == {
        "new": 5, "updated": 0, "deleted": 0, "unchanged": 0}
    assert sync_chunks(client, "test", chunks, embed, text_for_embedding)["unchanged"] == 5

    changed = chunks[:-1]  # last chunk removed...
    changed[0] = {**changed[0], "text": "Total assets 60,000", "content_hash": "new-hash"}  # ...first edited
    assert sync_chunks(client, "test", changed, embed, text_for_embedding) == {
        "new": 0, "updated": 1, "deleted": 1, "unchanged": 3}
    assert set(stored_hashes(client, "test")) == {c["chunk_id"] for c in changed}


def test_dense_search_with_filters(client):
    sync_chunks(client, "test", make_chunks(), embed, text_for_embedding)
    [(top, score)] = dense_search(client, "test", embed(["assets"])[0], limit=1)
    assert "assets" in top["text"].lower() and score > 0.9
    results = dense_search(client, "test", embed(["assets"])[0], limit=5,
                           filters={"company": "Costco", "doc_period": 2021})
    assert {c["company"] for c, _ in results} == {"Costco"}


def test_dimension_mismatch_is_detected(client):
    with pytest.raises(ValueError):
        ensure_collection(client, "test", dimension=384)
