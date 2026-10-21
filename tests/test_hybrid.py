import numpy as np
import pytest

pytest.importorskip("rank_bm25")

from finquery.evaluation.retrieval_eval import evaluate, filters_for, format_summary  # noqa: E402
from finquery.retrieval.fusion import rrf  # noqa: E402
from finquery.retrieval.memory_index import NumpyDenseIndex  # noqa: E402
from finquery.retrieval.retriever import Retriever  # noqa: E402


def chunk(cid, doc="DOC_A", page=1, text=""):
    return {"chunk_id": cid, "doc_name": doc, "company": doc, "doc_type": "10k", "doc_period": 2020,
            "page": page, "context": doc, "text": text or cid}


def test_rrf_rewards_agreement():
    a, b, c = chunk("a"), chunk("b"), chunk("c")
    fused = rrf([[(a, 0.9), (b, 0.8)], [(b, 12.0), (c, 11.0)]], k=60)
    assert [x["chunk_id"] for x, _ in fused] == ["b", "a", "c"]  # b is in both lists
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_numpy_index_search_and_filters():
    chunks = [chunk("a", "DOC_A"), chunk("b", "DOC_B")]
    index = NumpyDenseIndex(chunks, np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert index.search([0.0, 1.0], limit=1)[0][0]["chunk_id"] == "b"
    assert [c["chunk_id"] for c, _ in index.search([0.0, 1.0], 5, {"doc_name": "DOC_A"})] == ["a"]


VOCAB = ["assets", "revenue", "debt", "payable"]


class FakeEmbedder:
    def encode_query(self, text):
        v = np.array([text.lower().count(w) + 0.01 for w in VOCAB])
        return v / np.linalg.norm(v)


class FakeReranker:
    """Prefers chunks containing the word 'payable'."""
    def rerank(self, query, results, limit=None):
        scored = sorted(results, key=lambda r: "payable" in r[0]["text"], reverse=True)
        return [(c, float("payable" in c["text"])) for c, _ in scored][:limit]


def make_retriever():
    texts = {("DOC_A", 1): "total assets 100", ("DOC_A", 2): "revenue 50",
             ("DOC_A", 3): "accounts payable 30", ("DOC_B", 1): "total assets 999"}
    chunks = [chunk(f"{d}:{p}", d, p, t) for (d, p), t in texts.items()]
    vectors = np.array([FakeEmbedder().encode_query(c["text"]) for c in chunks])
    index = NumpyDenseIndex(chunks, vectors)
    return Retriever(chunks, FakeEmbedder(), index.search, FakeReranker(), candidates=4)


def test_retriever_modes():
    retriever = make_retriever()
    results = retriever.search_all_modes("total assets", k=2, filters={"doc_name": "DOC_B"})
    assert set(results) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    assert all(c["doc_name"] == "DOC_B" for rs in results.values() for c, _ in rs)
    top = retriever.search("revenue and accounts payable", k=1, mode="hybrid_rerank")
    assert top[0][0]["page"] == 3
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_evaluate_reports_recall_and_first_hit():
    question = {"financebench_id": "q1", "question_type": "metrics-generated", "doc_name": "DOC_A",
                "question": "accounts payable", "evidence": [{"doc_name": "DOC_A", "page": 3}]}
    report = evaluate(make_retriever(), [question], ks=(1, 4))
    run = report["questions"][0]["runs"]["document/hybrid_rerank"]
    assert run["first_hit"] == 1 and run["recall@1"] == 1
    assert report["summary"]["corpus/bm25"]["recall@4"] == 1
    assert "corpus/dense" in format_summary(report["summary"])
    assert filters_for(question, "corpus") is None
