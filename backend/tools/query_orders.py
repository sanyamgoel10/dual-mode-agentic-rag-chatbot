import sqlite3
from langsmith import traceable
from config import settings


@traceable(name="query_orders")
def query_orders(sql: str, db_path: str = None) -> dict:
    db_path = db_path or settings.DATABASE_PATH

    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        raise ValueError(f"Only SELECT statements are allowed. Got: {sql[:50]}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    return {"rows": rows, "sql": sql}
