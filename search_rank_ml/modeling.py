from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS
from .database import read_training_frame
from .metrics import add_ranks, auc_safe, ips_outcomes, policy_outcomes, ranking_metrics


@dataclass(frozen=True)
class TrainingResult:
    metrics_path: Path
    predictions_path: Path
    feature_importance_path: Path
    metrics: dict[str, object]


def train_and_evaluate(db_path: Path, model_dir: Path, metrics_path: Path) -> TrainingResult:
    model_dir.mkdir(parents=True, exist_ok=True)
    df = read_training_frame(db_path)
    score_min = df.groupby("session_id")["true_relevance_score"].transform("min")
    score_max = df.groupby("session_id")["true_relevance_score"].transform("max")
    df["relevance"] = ((df["true_relevance_score"] - score_min) / (score_max - score_min + 1e-9) * 3.0 + 0.65 * df["clicked"] + 1.25 * df["ordered"]).clip(0, 5)
    train_sessions, test_sessions = train_test_split(df["session_id"].drop_duplicates(), test_size=0.30, random_state=42)
    train_df = df.loc[df["session_id"].isin(train_sessions)].copy()
    test_df = df.loc[df["session_id"].isin(test_sessions)].copy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    x_test = scaler.transform(test_df[FEATURE_COLUMNS])

    click_model = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, max_leaf_nodes=24, random_state=42)
    click_model.fit(x_train, train_df["clicked"])
    cvr_model = RandomForestClassifier(n_estimators=140, max_depth=10, min_samples_leaf=12, random_state=42, n_jobs=-1)
    cvr_model.fit(x_train, train_df["ordered"])
    pairwise_model = _fit_pairwise_model(train_df, scaler)

    test_df["rule_model_score"] = test_df["rule_score"]
    test_df["pointwise_click_score"] = click_model.predict_proba(x_test)[:, 1]
    test_df["pointwise_cvr_score"] = cvr_model.predict_proba(x_test)[:, 1]
    test_df["pairwise_score"] = pairwise_model.decision_function(x_test)
    test_df["pairwise_score_norm"] = _minmax_by_session(test_df, "pairwise_score")
    test_df["rerank_score"] = (
        0.46 * test_df["pointwise_cvr_score"]
        + 0.22 * test_df["pointwise_click_score"]
        + 0.16 * test_df["pairwise_score_norm"]
        + 0.10 * _minmax_by_session(test_df, "expected_margin")
        + 0.06 * test_df["capacity_score"]
        - 0.035 * np.log1p(test_df["distance_km"])
    )

    rank_cols = {
        "rule": "rule_rank",
        "pointwise": "pointwise_rank",
        "pairwise": "pairwise_rank",
        "rerank": "rerank_rank",
        "logged": "logged_rank",
    }
    for score_col, rank_col in [
        ("rule_model_score", "rule_rank"),
        ("pointwise_cvr_score", "pointwise_rank"),
        ("pairwise_score", "pairwise_rank"),
        ("rerank_score", "rerank_rank"),
    ]:
        test_df = add_ranks(test_df, score_col, rank_col)

    model_metrics: dict[str, object] = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "sessions_test": int(test_df["session_id"].nunique()),
        "auc_click_pointwise": auc_safe(test_df["clicked"], test_df["pointwise_click_score"]),
        "auc_order_pointwise": auc_safe(test_df["ordered"], test_df["pointwise_cvr_score"]),
        "brier_click_pointwise": float(brier_score_loss(test_df["clicked"], test_df["pointwise_click_score"])),
        "brier_order_pointwise": float(brier_score_loss(test_df["ordered"], test_df["pointwise_cvr_score"])),
        "ranking": {},
        "policy_top3": {},
        "ips_top3": {},
    }

    for name, rank_col in rank_cols.items():
        model_metrics["ranking"][name] = ranking_metrics(test_df, rank_col)
        model_metrics["policy_top3"][name] = policy_outcomes(test_df, rank_col)
        model_metrics["ips_top3"][name] = ips_outcomes(test_df, rank_col)

    rule_ndcg = model_metrics["ranking"]["rule"]["ndcg@10"]
    rerank_ndcg = model_metrics["ranking"]["rerank"]["ndcg@10"]
    model_metrics["ndcg10_lift_vs_rule"] = float((rerank_ndcg / max(rule_ndcg, 1e-9)) - 1.0)
    model_metrics["feature_importance"] = _permutation_importance(test_df, FEATURE_COLUMNS, "rerank_score")

    predictions_path = model_dir / "ranking_predictions.csv"
    feature_importance_path = model_dir / "feature_importance.csv"
    test_df[
        [
            "session_id",
            "query_id",
            "merchant_id",
            "variant",
            "logged_rank",
            "rule_rank",
            "pointwise_rank",
            "pairwise_rank",
            "rerank_rank",
            "clicked",
            "ordered",
            "gross_revenue",
            "gross_margin",
            "rule_model_score",
            "pointwise_cvr_score",
            "pairwise_score",
            "rerank_score",
        ]
    ].to_csv(predictions_path, index=False)
    pd.DataFrame(model_metrics["feature_importance"]).to_csv(feature_importance_path, index=False)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(model_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainingResult(metrics_path, predictions_path, feature_importance_path, model_metrics)


def _fit_pairwise_model(train_df: pd.DataFrame, scaler: StandardScaler) -> LogisticRegression:
    rows = []
    labels = []
    for _, group in train_df.groupby("session_id", sort=False):
        positives = group.loc[group["relevance"] > 0]
        negatives = group.loc[group["relevance"] == 0]
        if positives.empty or negatives.empty:
            continue
        pos = positives.nlargest(1, "relevance")[FEATURE_COLUMNS].iloc[0].to_numpy(dtype=float)
        sampled = negatives.sample(n=min(2, len(negatives)), random_state=17)
        for _, neg_row in sampled.iterrows():
            neg = neg_row[FEATURE_COLUMNS].to_numpy(dtype=float)
            rows.append(pos - neg)
            labels.append(1)
            rows.append(neg - pos)
            labels.append(0)
    if not rows:
        rows = [np.zeros(len(FEATURE_COLUMNS)), np.ones(len(FEATURE_COLUMNS))]
        labels = [0, 1]
    x_pair = scaler.transform(pd.DataFrame(rows, columns=FEATURE_COLUMNS))
    model = LogisticRegression(max_iter=800, class_weight="balanced", random_state=42)
    model.fit(x_pair, labels)
    return model


def _minmax_by_session(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("session_id")[col]
    min_v = grouped.transform("min")
    max_v = grouped.transform("max")
    return (df[col] - min_v) / (max_v - min_v + 1e-9)


def _permutation_importance(df: pd.DataFrame, feature_cols: list[str], score_col: str) -> list[dict[str, object]]:
    base = ranking_metrics(add_ranks(df, score_col, "_tmp_rank"), "_tmp_rank")["ndcg@10"]
    rows: list[dict[str, float | str]] = []
    rng = np.random.default_rng(42)
    for col in feature_cols:
        shuffled = df.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        synthetic_score = df[score_col] - 0.05 * df[col] + 0.05 * shuffled[col]
        shuffled["_score"] = synthetic_score
        shuffled = add_ranks(shuffled, "_score", "_rank")
        ndcg = ranking_metrics(shuffled, "_rank")["ndcg@10"]
        rows.append({"feature": col, "importance": round(float(base - ndcg), 6)})
    return sorted(rows, key=lambda item: float(item["importance"]), reverse=True)
