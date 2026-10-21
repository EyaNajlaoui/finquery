from collections import Counter

import pytest

from finquery.config import get_settings
from finquery.data.financebench import merge_documents, normalize, select, stratified_split


def make_questions():
    types = ["novel-generated"] * 27 + ["domain-relevant"] * 16 + ["metrics-generated"] * 7
    return [{"financebench_id": f"id_{i:03d}", "doc_name": f"DOC_{i % 5}", "question_type": t}
            for i, t in enumerate(types)]


def test_split_sizes_and_proportions():
    dev, test = stratified_split(make_questions(), dev_size=20, seed=42)
    assert len(dev) == 20 and len(test) == 30
    assert Counter(q["question_type"] for q in dev) == {
        "novel-generated": 11, "domain-relevant": 6, "metrics-generated": 3}


def test_split_is_reproducible_and_disjoint():
    dev1, test1 = stratified_split(make_questions(), dev_size=20, seed=42)
    dev2, _ = stratified_split(make_questions(), dev_size=20, seed=42)
    assert [q["financebench_id"] for q in dev1] == [q["financebench_id"] for q in dev2]
    assert not {q["financebench_id"] for q in dev1} & {q["financebench_id"] for q in test1}


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        stratified_split(make_questions(), dev_size=0, seed=1)


def test_normalize_converts_pages_to_one_based():
    raw = {"financebench_id": "x", "doc_name": "AMD_2022_10K", "question_type": "novel-generated",
           "evidence": [{"evidence_text": "Total current liabilities", "doc_name": "AMD_2022_10K",
                         "evidence_page_num": 55}]}
    row = normalize(raw)
    assert row["evidence_pages"] == [56]
    assert row["evidence"][0]["page"] == 56


def test_merge_and_select():
    questions = [{"doc_name": "A", "question": "q1"}, {"doc_name": "B", "question": "q2"}]
    merged = merge_documents(questions, [{"doc_name": "A", "doc_type": "10k"}])
    assert merged[0]["doc_type"] == "10k"
    assert select(merged, ["A"]) == [merged[0]]


def test_settings_list_18_documents():
    assert len(get_settings()["data"]["documents"]) == 18


def test_document_catalog():
    from finquery.data.financebench import document_catalog
    rows = [{"doc_name": "B", "company": "b", "doc_type": "10k", "doc_period": 2020, "question": "x"},
            {"doc_name": "A", "company": "a", "doc_type": "8k", "doc_period": 2021, "question": "y"},
            {"doc_name": "A", "company": "a", "doc_type": "8k", "doc_period": 2021, "question": "z"}]
    assert document_catalog(rows) == [
        {"doc_name": "A", "company": "a", "doc_type": "8k", "doc_period": 2021},
        {"doc_name": "B", "company": "b", "doc_type": "10k", "doc_period": 2020}]
