from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_TABLE_DIR, SQL_DIR
from .database import read_sql
from .plotting import create_figures
from .reporting import write_model_card, write_report


SQL_FILES = {
    "funnel": "01_search_funnel.sql",
    "supply": "02_query_supply_coverage.sql",
    "position_bias": "03_position_bias.sql",
    "experiment": "04_experiment_lift.sql",
    "merchant_quality": "05_merchant_quality.sql",
    "tail_performance": "06_tail_query_performance.sql",
    "feature_dataset": "07_feature_dataset.sql",
    "bad_cases": "08_bad_case_queries.sql",
}


def run_sql_suite(db_path: Path) -> dict[str, pd.DataFrame]:
    outputs = {}
    for name, filename in SQL_FILES.items():
        df = read_sql(db_path, (SQL_DIR / filename).read_text(encoding="utf-8"))
        if df.empty:
            raise ValueError(f"SQL returned empty result: {filename}")
        outputs[name] = df
    return outputs


def write_analysis_outputs(db_path: Path, metrics_path: Path, report_path: Path, model_card_path: Path, figure_dir: Path) -> dict[str, int]:
    results = run_sql_suite(db_path)
    DEFAULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(DEFAULT_TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    figures = create_figures(results, metrics_path, figure_dir)
    write_report(results, metrics_path, figures, report_path)
    write_model_card(results, metrics_path, model_card_path)
    return {name: len(df) for name, df in results.items()}
