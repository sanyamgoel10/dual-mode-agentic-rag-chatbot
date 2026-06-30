import sqlite3
import pandas as pd
import os
from config import settings


def init_db(csv_path: str = None, db_path: str = None) -> None:
    csv_path = csv_path or settings.ORDERS_CSV_PATH
    db_path = db_path or settings.DATABASE_PATH

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Orders DB initialised at {settings.DATABASE_PATH}")
