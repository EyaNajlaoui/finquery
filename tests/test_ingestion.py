import pytest

from finquery.evaluation.metrics import token_coverage
from finquery.ingestion.chunker import chunk_page, page_title, split_lines, text_for_embedding
from finquery.ingestion.parser import clean_pages

DOC = {"doc_name": "COSTCO_2021_10K", "company": "Costco", "doc_type": "10k", "doc_period": 2021}


def test_clean_pages_removes_repeated_headers_and_page_numbers():
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    pages = [f"Table of Contents\n{w} title\nBody\nNet sales 100\n{w} note\n{i}"
             for i, w in enumerate(words, start=1)]
    cleaned = clean_pages(pages)
    assert cleaned[0] == "alpha title\nBody\nNet sales 100\nalpha note"
    assert all("Table of Contents" not in p for p in cleaned)


def test_clean_pages_keeps_short_pages_intact():
    assert clean_pages(["Net sales 100\n1"] * 5) == ["Net sales 100\n1"] * 5


def test_clean_pages_keeps_repeated_lines_inside_the_page():
    # "Net sales" is in the middle of every page: it must stay
    pages = [f"Title {i}\nIntro\nNet sales 100\nCost 50\nEnd {i}\nfooter {i}" for i in range(5)]
    assert all("Net sales 100" in p for p in clean_pages(pages))


def test_page_title():
    assert page_title("COSTCO WHOLESALE CORPORATION\nCONSOLIDATED BALANCE SHEETS\n(in millions)") == \
        "COSTCO WHOLESALE CORPORATION - CONSOLIDATED BALANCE SHEETS"
    assert page_title("Revenue grew in 2021.\nSome text") == ""


def test_split_lines_respects_size_and_never_cuts_a_line():
    lines = [f"Line {i} " + "word " * 8 for i in range(30)]  # 10 words per line
    pieces = split_lines("\n".join(lines), max_words=50, overlap_words=10)
    assert all(len(p.split()) <= 50 for p in pieces)
    # every original line appears whole in at least one piece
    assert set().union(*(p.splitlines() for p in pieces)) == {line.strip() for line in lines}


def test_split_lines_overlap():
    lines = [f"row{i} a b c d" for i in range(20)]  # 5 words per line
    first, second = split_lines("\n".join(lines), max_words=20, overlap_words=5)[:2]
    assert first.splitlines()[-1] == second.splitlines()[0]


def test_split_lines_cuts_very_long_lines():
    pieces = split_lines(" ".join(f"w{i}" for i in range(250)), max_words=100, overlap_words=20)
    assert len(pieces) == 3 and all(len(p.split()) <= 100 for p in pieces)


def test_split_lines_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_lines("a b c", max_words=10, overlap_words=10)


def test_chunk_page_metadata_and_context():
    text = "CONSOLIDATED BALANCE SHEETS\nTotal current assets 29,505 28,120"
    [chunk] = chunk_page(text, page=38, doc=DOC, max_words=150, overlap_words=30)
    assert chunk["chunk_id"] == "COSTCO_2021_10K:p38:c0"
    assert chunk["page"] == 38 and chunk["company"] == "Costco"
    assert chunk["context"] == "Costco | 10-K | FY2021 | page 38 | CONSOLIDATED BALANCE SHEETS"
    assert text_for_embedding(chunk).startswith("Costco | 10-K")


def test_content_hash_changes_with_text():
    a = chunk_page("Net sales 100", 1, DOC, 150, 30)[0]["content_hash"]
    b = chunk_page("Net sales 101", 1, DOC, 150, 30)[0]["content_hash"]
    assert a != b


def test_token_coverage():
    assert token_coverage("Total assets $59,268", "TOTAL ASSETS $ 59,268 $ 55,556") == 1
    assert token_coverage("net income 3,033", "net sales 3,033") == pytest.approx(2 / 3)
