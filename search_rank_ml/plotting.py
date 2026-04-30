from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-search-rank")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/matplotlib-search-rank-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Heiti TC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


def create_figures(results: dict[str, pd.DataFrame], metrics_path: Path, figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    figures = {
        "funnel": _plot_funnel(results["funnel"], figure_dir / "01_search_funnel.png"),
        "experiment": _plot_experiment(results["experiment"], figure_dir / "02_experiment_lift.png"),
        "ranking": _plot_ranking_metrics(metrics, figure_dir / "03_ranking_metrics.png"),
        "ips": _plot_ips(metrics, figure_dir / "04_ips_counterfactual.png"),
        "tail": _plot_tail(results["tail_performance"], figure_dir / "05_tail_query_performance.png"),
        "position_bias": _plot_position_bias(results["position_bias"], figure_dir / "06_position_bias.png"),
        "features": _plot_features(metrics, figure_dir / "07_feature_importance.png"),
        "bad_cases": _plot_bad_cases(results["bad_cases"], figure_dir / "08_bad_case_queries.png"),
    }
    return figures


def _plot_funnel(df: pd.DataFrame, path: Path) -> Path:
    data = df.copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(data["stage"], data["users"], color="#33658a")
    ax.set_title("Search conversion funnel")
    ax.set_ylabel("Users / sessions")
    ax.tick_params(axis="x", labelrotation=25)
    for idx, row in data.iterrows():
        ax.text(idx, row["users"] * 1.01, f"{int(row['users']):,}", ha="center", fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_experiment(df: pd.DataFrame, path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.bar(df["variant"], df["order_rate"], color=["#8d99ae", "#457b9d", "#2a9d8f"])
    ax1.set_ylabel("Order rate")
    ax2 = ax1.twinx()
    ax2.plot(df["variant"], df["gmv_per_session"], color="#e76f51", marker="o")
    ax2.set_ylabel("GMV per session")
    ax1.set_title("A/B experiment lift")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_ranking_metrics(metrics: dict[str, object], path: Path) -> Path:
    ranking = metrics["ranking"]
    models = ["rule", "pointwise", "pairwise", "rerank"]
    ndcg = [ranking[name]["ndcg@10"] for name in models]
    mrr = [ranking[name]["mrr"] for name in models]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = range(len(models))
    ax.bar([i - 0.18 for i in x], ndcg, width=0.36, label="NDCG@10", color="#2a9d8f")
    ax.bar([i + 0.18 for i in x], mrr, width=0.36, label="MRR", color="#e9c46a")
    ax.set_xticks(list(x), models)
    ax.set_ylim(0, max(ndcg + mrr) * 1.25)
    ax.set_title("Offline ranking quality")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_ips(metrics: dict[str, object], path: Path) -> Path:
    ips = metrics["ips_top3"]
    models = ["rule", "pointwise", "pairwise", "rerank"]
    gmv = [ips[name]["ips_gmv_per_session"] for name in models]
    orders = [ips[name]["ips_orders_per_session"] for name in models]
    fig, ax1 = plt.subplots(figsize=(7.8, 4.8))
    ax1.bar(models, gmv, color="#517664")
    ax1.set_ylabel("IPS GMV / session")
    ax2 = ax1.twinx()
    ax2.plot(models, orders, color="#b02e0c", marker="o")
    ax2.set_ylabel("IPS orders / session")
    ax1.set_title("Counterfactual policy evaluation")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_tail(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(df["tail_type"], df["order_rate"], color="#5d7fba")
    ax.set_title("Tail query order rate")
    ax.set_ylabel("Order rate")
    for idx, row in df.iterrows():
        ax.text(idx, row["order_rate"] * 1.03, f"{row['order_rate']:.1%}", ha="center", fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_position_bias(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(df["logged_rank"], df["click_rate"], marker="o", label="CTR")
    ax.plot(df["logged_rank"], df["order_rate"], marker="o", label="CVR")
    ax.set_title("Logged position bias")
    ax.set_xlabel("Logged rank")
    ax.set_ylabel("Rate")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_features(metrics: dict[str, object], path: Path) -> Path:
    data = pd.DataFrame(metrics["feature_importance"]).sort_values("importance", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(data["feature"][::-1], data["importance"][::-1], color="#713e5a")
    ax.set_title("Reranker feature sensitivity")
    ax.set_xlabel("NDCG@10 drop")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_bad_cases(df: pd.DataFrame, path: Path) -> Path:
    data = df.sort_values("order_rate").head(10).copy()
    label = data["intent_category"] + " / " + data["tail_type"]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(label[::-1], data["order_rate"][::-1], color="#bf7d3a")
    ax.set_title("Low conversion query segments")
    ax.set_xlabel("Order rate")
    fig.savefig(path)
    plt.close(fig)
    return path
