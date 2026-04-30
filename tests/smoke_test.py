from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from search_rank_ml.analysis import SQL_FILES, write_analysis_outputs  # noqa: E402
from search_rank_ml.config import SQL_DIR  # noqa: E402
from search_rank_ml.data_generation import generate_synthetic_data  # noqa: E402
from search_rank_ml.database import build_database  # noqa: E402
from search_rank_ml.modeling import train_and_evaluate  # noqa: E402


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        csv_dir = root / "data"
        db_path = root / "search.sqlite"
        metrics_path = root / "output" / "metrics.json"
        model_dir = root / "output" / "models"
        report_path = root / "reports" / "case.md"
        model_card = root / "reports" / "model_card.md"
        figures = root / "reports" / "figures"

        generated = generate_synthetic_data(csv_dir, n_users=800, n_merchants=160, n_queries=70, n_sessions=1200, seed=7)
        assert generated.row_counts["impressions"] == 12000
        _assert_privacy(csv_dir)
        _assert_quality(csv_dir)

        build_database(csv_dir, db_path)
        _assert_sql(db_path)

        train = train_and_evaluate(db_path, model_dir, metrics_path)
        metrics = json.loads(train.metrics_path.read_text(encoding="utf-8"))
        assert metrics["ranking"]["rerank"]["ndcg@10"] > metrics["ranking"]["rule"]["ndcg@10"] * 1.03
        assert metrics["auc_click_pointwise"] >= 0.60
        assert metrics["ips_top3"]["rerank"]["ips_gmv_per_session"] > 0

        counts = write_analysis_outputs(db_path, metrics_path, report_path, model_card, figures)
        assert counts["funnel"] > 0
        assert report_path.exists()
        assert model_card.exists()
        assert len(list(figures.glob("*.png"))) == 8
        print("smoke_test passed")


def _assert_quality(csv_dir: Path) -> None:
    sessions = pd.read_csv(csv_dir / "search_sessions.csv")
    impressions = pd.read_csv(csv_dir / "impressions.csv")
    orders = pd.read_csv(csv_dir / "orders.csv")
    assert impressions["session_id"].isin(sessions["session_id"]).all()
    assert orders["session_id"].isin(sessions["session_id"]).all()
    assert impressions[["distance_km", "rating", "logged_propensity", "gross_revenue", "gross_margin"]].ge(0).all().all()
    assert impressions["logged_rank"].between(1, 10).all()
    shares = impressions.drop_duplicates("session_id")["variant"].value_counts(normalize=True)
    assert 0.10 <= shares["holdout"] <= 0.20
    assert 0.35 <= shares["control"] <= 0.50
    assert 0.35 <= shares["treatment"] <= 0.50


def _assert_sql(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        for filename in SQL_FILES.values():
            df = pd.read_sql_query((SQL_DIR / filename).read_text(encoding="utf-8"), conn)
            assert not df.empty, filename


def _assert_privacy(csv_dir: Path) -> None:
    forbidden = ["安" + "吉" + "星", "上" + "汽", "On" + "Star", "BD" + "CO", "真实" + "门店", "真实" + "交易", "手机" + "号", "V" + "IN"]
    for path in csv_dir.glob("*.csv"):
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{term} found in {path}"


if __name__ == "__main__":
    main()
