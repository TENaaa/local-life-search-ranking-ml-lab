from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


TABLES = ["users", "queries", "merchants", "search_sessions", "experiments", "impressions", "orders"]


def build_database(csv_dir: Path, db_path: Path) -> dict[str, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        for table in TABLES:
            path = csv_dir / f"{table}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing CSV: {path}")
            df = pd.read_csv(path)
            df.to_sql(table, conn, index=False, if_exists="replace")
            counts[table] = len(df)
        conn.executescript(
            """
            CREATE INDEX idx_sessions_user ON search_sessions(user_id);
            CREATE INDEX idx_sessions_query ON search_sessions(query_id);
            CREATE INDEX idx_impressions_session ON impressions(session_id);
            CREATE INDEX idx_impressions_query ON impressions(query_id);
            CREATE INDEX idx_impressions_merchant ON impressions(merchant_id);
            CREATE INDEX idx_impressions_variant ON impressions(variant);
            CREATE INDEX idx_orders_session ON orders(session_id);
            CREATE INDEX idx_experiments_session ON experiments(session_id);
            """
        )
    return counts


def read_sql(db_path: Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def read_training_frame(db_path: Path) -> pd.DataFrame:
    query = """
    SELECT
        i.*,
        q.intent_category,
        q.tail_type,
        s.city_tier,
        s.hour,
        m.merchant_category,
        m.price_level
    FROM impressions i
    JOIN queries q ON q.query_id = i.query_id
    JOIN search_sessions s ON s.session_id = i.session_id
    JOIN merchants m ON m.merchant_id = i.merchant_id
    """
    return read_sql(db_path, query)
