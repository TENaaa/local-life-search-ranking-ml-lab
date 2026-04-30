from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_DB_PATH = PROJECT_ROOT / "output" / "search_ranking.sqlite"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "output" / "models"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "output" / "ranking_metrics.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "search_ranking_ml_case.md"
DEFAULT_MODEL_CARD_PATH = PROJECT_ROOT / "reports" / "model_card.md"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
SQL_DIR = PROJECT_ROOT / "sql"

FEATURE_COLUMNS = [
    "distance_km",
    "rating",
    "price_match_score",
    "category_match",
    "service_quality",
    "capacity_score",
    "is_open",
    "merchant_historical_ctr",
    "merchant_historical_cvr",
    "query_tail_score",
    "query_urgency_score",
    "user_active_score",
    "rank_position_signal",
    "expected_margin",
]
