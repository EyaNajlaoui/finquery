import pytest

from finquery.sql.safety import check_sql, clean_sql


def test_select_is_accepted():
    assert check_sql("SELECT name FROM companies WHERE year = 2024") is None
    assert check_sql("WITH t AS (SELECT 1) SELECT * FROM t") is None


@pytest.mark.parametrize("sql", [
    "DELETE FROM companies",
    "SELECT 1; DROP TABLE companies",
    "SELECT * FROM companies WHERE 1=1 UNION SELECT * FROM x; UPDATE x SET a=1",
    "PRAGMA table_info(companies)",
    "INSERT INTO companies VALUES (1)",
])
def test_dangerous_queries_are_rejected(sql):
    assert check_sql(sql) is not None


def test_clean_sql_removes_fences_and_semicolon():
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
