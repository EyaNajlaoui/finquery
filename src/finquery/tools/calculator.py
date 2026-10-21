"""Safe arithmetic for the agent's `calculate` tool.

The expression is parsed with `ast` and only numbers and arithmetic operators are
evaluated. Unlike eval(), no other Python code can run.
"""
from __future__ import annotations

import ast
import operator

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
MAX_EXPONENT = 100


class CalculatorError(ValueError):
    pass


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise CalculatorError("only numbers and + - * / % ** ( ) are allowed")


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression. Accepts 1,234.5 thousands separators."""
    cleaned = expression.replace(",", "").replace("_", "").strip()
    if not cleaned:
        raise CalculatorError("empty expression")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as error:
        raise CalculatorError(f"invalid expression: {error.msg}") from None
    try:
        return round(float(_evaluate(tree.body)), 6)
    except ZeroDivisionError:
        raise CalculatorError("division by zero") from None
