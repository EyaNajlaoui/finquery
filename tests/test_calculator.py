import pytest

from finquery.tools.calculator import CalculatorError, calculate


def test_basic_arithmetic():
    assert calculate("(55900 - 48200) / 48200 * 100") == pytest.approx(15.975104)
    assert calculate("2 ** 10") == 1024


def test_thousands_separators_are_accepted():
    assert calculate("1,234.5 + 0.5") == 1235


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hacked')",
    "open('secrets.txt')",
    "[1, 2, 3]",
    "x + 1",
    "True + 1",
])
def test_code_is_rejected(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_division_by_zero():
    with pytest.raises(CalculatorError):
        calculate("1 / 0")


def test_huge_exponent_is_rejected():
    with pytest.raises(CalculatorError):
        calculate("10 ** 100000")
