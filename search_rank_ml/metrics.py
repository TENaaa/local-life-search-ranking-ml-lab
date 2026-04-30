from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def add_ranks(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    out = df.copy()
    out[rank_col] = out.groupby("session_id")[score_col].rank(method="first", ascending=False).astype(int)
    return out


def ranking_metrics(df: pd.DataFrame, rank_col: str, label_col: str = "relevance") -> dict[str, float]:
    grouped = df.groupby("session_id", sort=False)
    return {
        "ndcg@5": float(np.mean([_ndcg(group, rank_col, label_col, 5) for _, group in grouped])),
        "ndcg@10": float(np.mean([_ndcg(group, rank_col, label_col, 10) for _, group in grouped])),
        "map@10": float(np.mean([_average_precision(group, rank_col, 10) for _, group in grouped])),
        "mrr": float(np.mean([_reciprocal_rank(group, rank_col) for _, group in grouped])),
    }


def auc_safe(y_true: pd.Series, y_score: pd.Series) -> float:
    if y_true.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def policy_outcomes(df: pd.DataFrame, rank_col: str, k: int = 3) -> dict[str, float]:
    chosen = df.loc[df[rank_col] <= k].copy()
    session_count = max(1, df["session_id"].nunique())
    return {
        "topk_ctr": float(chosen["clicked"].sum() / session_count),
        "topk_cvr": float(chosen["ordered"].sum() / session_count),
        "topk_gmv": float(chosen["gross_revenue"].sum() / session_count),
        "topk_margin": float(chosen["gross_margin"].sum() / session_count),
    }


def ips_outcomes(df: pd.DataFrame, rank_col: str, k: int = 3) -> dict[str, float]:
    chosen = df.loc[df[rank_col] <= k].copy()
    weights = np.minimum(1.0 / chosen["logged_propensity"].clip(0.04, 1.0), 10.0)
    denom = max(1.0, float(df["session_id"].nunique()))
    return {
        "ips_clicks_per_session": float((chosen["clicked"] * weights).sum() / denom),
        "ips_orders_per_session": float((chosen["ordered"] * weights).sum() / denom),
        "ips_gmv_per_session": float((chosen["gross_revenue"] * weights).sum() / denom),
        "ips_margin_per_session": float((chosen["gross_margin"] * weights).sum() / denom),
    }


def _ndcg(group: pd.DataFrame, rank_col: str, label_col: str, k: int) -> float:
    ordered = group.sort_values(rank_col).head(k)
    gains = (np.power(2, ordered[label_col].to_numpy()) - 1) / np.log2(np.arange(2, len(ordered) + 2))
    ideal = group.sort_values(label_col, ascending=False).head(k)
    ideal_gains = (np.power(2, ideal[label_col].to_numpy()) - 1) / np.log2(np.arange(2, len(ideal) + 2))
    denom = ideal_gains.sum()
    return float(gains.sum() / denom) if denom > 0 else 0.0


def _average_precision(group: pd.DataFrame, rank_col: str, k: int) -> float:
    ordered = group.sort_values(rank_col).head(k)
    relevant = (ordered["ordered"] > 0) | (ordered["clicked"] > 0)
    if relevant.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for idx, is_rel in enumerate(relevant, start=1):
        if is_rel:
            hits += 1
            precisions.append(hits / idx)
    return float(np.mean(precisions)) if precisions else 0.0


def _reciprocal_rank(group: pd.DataFrame, rank_col: str) -> float:
    ordered = group.sort_values(rank_col)
    relevant = ordered.loc[(ordered["ordered"] > 0) | (ordered["clicked"] > 0)]
    if relevant.empty:
        return 0.0
    return float(1.0 / int(relevant.iloc[0][rank_col]))
