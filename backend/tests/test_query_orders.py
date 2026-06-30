import pytest
import os
from startup import init_db
from tools.query_orders import query_orders

DB_PATH = "/tmp/test_query.db"
CSV_PATH = "tests/fixtures/orders_sample.csv"


def setup_module():
    init_db(CSV_PATH, DB_PATH)


def teardown_module():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


def test_select_returns_rows():
    result = query_orders("SELECT * FROM orders WHERE status='pending'", DB_PATH)
    assert result["sql"] == "SELECT * FROM orders WHERE status='pending'"
    assert len(result["rows"]) == 1
    assert result["rows"][0]["order_id"] == "ORD-0002"


def test_count_query():
    result = query_orders("SELECT COUNT(*) as total FROM orders", DB_PATH)
    assert result["rows"][0]["total"] == 5


def test_sum_query():
    result = query_orders(
        "SELECT SUM(amount) as revenue FROM orders WHERE status='delivered'", DB_PATH
    )
    assert result["rows"][0]["revenue"] == pytest.approx(1899.0 + 7497.0)


def test_non_select_raises():
    with pytest.raises(ValueError, match="Only SELECT"):
        query_orders("DROP TABLE orders", DB_PATH)


def test_delete_raises():
    with pytest.raises(ValueError, match="Only SELECT"):
        query_orders("DELETE FROM orders WHERE order_id='ORD-0001'", DB_PATH)


def test_empty_result():
    result = query_orders("SELECT * FROM orders WHERE status='nonexistent'", DB_PATH)
    assert result["rows"] == []
