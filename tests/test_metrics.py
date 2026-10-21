import pytest

from finquery.evaluation.metrics import contains_number, extract_numbers, recall_at_k, reciprocal_rank


def test_recall_at_k():
    assert recall_at_k([5, 9, 12, 3], [9], k=4) == 1
    assert recall_at_k([5, 9, 12, 3], [9, 40], k=4) == 0.5
    assert recall_at_k([5, 9, 12, 3], [3], k=2) == 0


def test_reciprocal_rank():
    assert reciprocal_rank([5, 9, 12], [9], k=4) == 0.5
    assert reciprocal_rank([5, 9, 12], [40], k=4) == 0


def test_recall_requires_relevant_pages():
    with pytest.raises(ValueError):
        recall_at_k([1], [], k=4)


def test_extract_numbers():
    assert extract_numbers("Revenue was $1,234.5 million, up 15.98% (-3 pts)") == [1234.5, 15.98, -3.0]


def test_contains_number_accepts_percentage_form_of_ratios():
    assert contains_number("The quick ratio is 1.57.", 1.57)
    assert contains_number("Debt-to-equity is 61.99%.", 0.62)
    assert not contains_number("Revenue grew 14.1%.", 15.98)
