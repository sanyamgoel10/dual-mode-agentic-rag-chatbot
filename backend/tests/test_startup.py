import sqlite3
import os
import pytest
from startup import init_db

DB_PATH = "/tmp/test_startup.db"
CSV_PATH = "tests/fixtures/orders_sample.csv"


def teardown_function():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


def test_init_db_creates_table():
    init_db(CSV_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='orders'")
    assert cursor.fetchone() is not None
    conn.close()


def test_init_db_loads_rows():
    init_db(CSV_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert count == 5
    conn.close()


def test_init_db_correct_columns():
    init_db(CSV_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT order_id, customer, product, amount, status, order_date FROM orders LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row[0] == "ORD-0001"
    assert row[3] == 1899.0
    conn.close()


def test_init_db_idempotent():
    init_db(CSV_PATH, DB_PATH)
    init_db(CSV_PATH, DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert count == 5
    conn.close()
