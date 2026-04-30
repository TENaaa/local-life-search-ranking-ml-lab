from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataGenerationResult:
    output_dir: Path
    row_counts: dict[str, int]


CATEGORIES = ["car_wash", "tire_repair", "chauffeur", "maintenance", "restaurant", "parent_child"]
QUERY_TEMPLATES = {
    "car_wash": ["洗车", "附近洗车", "精洗", "夜间洗车", "新能源洗车"],
    "tire_repair": ["补胎", "轮胎修补", "附近补胎", "应急补胎", "轮胎检查"],
    "chauffeur": ["代驾", "夜间代驾", "商务代驾", "附近代驾", "长途代驾"],
    "maintenance": ["保养", "小保养", "空调清洗", "机油保养", "刹车检查"],
    "restaurant": ["家庭聚餐", "工作餐", "火锅", "轻食", "夜宵"],
    "parent_child": ["亲子餐厅", "儿童乐园", "周末亲子", "亲子活动", "儿童餐"],
}


def generate_synthetic_data(
    output_dir: Path,
    n_users: int = 4500,
    n_merchants: int = 720,
    n_queries: int = 240,
    n_sessions: int = 9000,
    candidates_per_session: int = 10,
    seed: int = 42,
) -> DataGenerationResult:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.csv"):
        stale.unlink()

    users = _make_users(rng, n_users)
    queries = _make_queries(rng, n_queries)
    merchants = _make_merchants(rng, n_merchants)
    sessions = _make_sessions(rng, users, queries, n_sessions)
    experiments = _make_experiments(sessions, rng)
    impressions, orders = _make_impressions_and_orders(rng, users, queries, merchants, sessions, experiments, candidates_per_session)

    tables = {
        "users": users,
        "queries": queries,
        "merchants": merchants,
        "search_sessions": sessions,
        "experiments": experiments,
        "impressions": impressions,
        "orders": orders,
    }
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    return DataGenerationResult(output_dir, {name: len(df) for name, df in tables.items()})


def _make_users(rng: np.random.Generator, n: int) -> pd.DataFrame:
    prefs = rng.choice(CATEGORIES, n, p=[0.17, 0.13, 0.14, 0.18, 0.23, 0.15])
    active = rng.choice(["low", "mid", "high"], n, p=[0.30, 0.46, 0.24])
    city = rng.choice(["T1", "T2", "T3", "T4"], n, p=[0.18, 0.34, 0.31, 0.17])
    zone_x = rng.integers(0, 24, n)
    zone_y = rng.integers(0, 24, n)
    return pd.DataFrame(
        {
            "user_id": [f"U{i:06d}" for i in range(1, n + 1)],
            "city_tier": city,
            "home_zone_x": zone_x,
            "home_zone_y": zone_y,
            "price_sensitivity": rng.beta(2.3, 2.8, n).round(4),
            "preferred_category": prefs,
            "app_active_level": active,
            "historical_orders_90d": rng.poisson(np.select([active == "low", active == "mid"], [0.7, 1.8], default=4.2)).clip(0, 18),
            "user_active_score": np.select([active == "low", active == "mid"], [0.35, 0.62], default=0.88).round(3),
        }
    )


def _make_queries(rng: np.random.Generator, n: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        category = str(rng.choice(CATEGORIES, p=[0.17, 0.13, 0.14, 0.18, 0.23, 0.15]))
        base = str(rng.choice(QUERY_TEMPLATES[category]))
        tail_type = str(rng.choice(["head", "mid", "tail"], p=[0.28, 0.44, 0.28]))
        modifier = "" if tail_type == "head" else str(rng.choice([" 高评分", " 优惠", " 现在营业", " 近一点", " 周末可用", " 不排队"]))
        urgency = float(np.clip(rng.normal(0.68 if category in {"tire_repair", "chauffeur"} else 0.45, 0.18), 0.05, 0.98))
        rows.append(
            {
                "query_id": f"Q{i:05d}",
                "query_text": base + modifier,
                "intent_category": category,
                "tail_type": tail_type,
                "query_urgency_score": round(urgency, 4),
                "query_tail_score": {"head": 0.15, "mid": 0.45, "tail": 0.85}[tail_type],
                "avg_order_value": round(float(rng.normal(_category_aov(category), _category_aov(category) * 0.18)), 2),
            }
        )
    return pd.DataFrame(rows)


def _make_merchants(rng: np.random.Generator, n: int) -> pd.DataFrame:
    category = rng.choice(CATEGORIES, n, p=[0.17, 0.13, 0.14, 0.18, 0.23, 0.15])
    quality = np.clip(rng.beta(5, 2, n), 0.1, 0.99)
    rating = np.round(3.2 + quality * 1.7 + rng.normal(0, 0.13, n), 2).clip(3.0, 5.0)
    price_level = rng.choice(["low", "mid", "high"], n, p=[0.32, 0.48, 0.20])
    return pd.DataFrame(
        {
            "merchant_id": [f"M{i:05d}" for i in range(1, n + 1)],
            "merchant_category": category,
            "city_tier": rng.choice(["T1", "T2", "T3", "T4"], n, p=[0.20, 0.35, 0.30, 0.15]),
            "zone_x": rng.integers(0, 24, n),
            "zone_y": rng.integers(0, 24, n),
            "rating": rating,
            "price_level": price_level,
            "avg_price": [_price_for_category(cat, lvl, rng) for cat, lvl in zip(category, price_level)],
            "service_quality": quality.round(4),
            "capacity_score": rng.beta(4, 2.4, n).round(4),
            "open_probability": rng.uniform(0.70, 0.98, n).round(4),
            "merchant_historical_ctr": np.clip(0.05 + quality * 0.22 + rng.normal(0, 0.025, n), 0.01, 0.45).round(4),
            "merchant_historical_cvr": np.clip(0.015 + quality * 0.11 + rng.normal(0, 0.015, n), 0.003, 0.25).round(4),
        }
    )


def _make_sessions(rng: np.random.Generator, users: pd.DataFrame, queries: pd.DataFrame, n: int) -> pd.DataFrame:
    sampled_users = users.sample(n=n, replace=True, random_state=int(rng.integers(1, 1_000_000))).reset_index(drop=True)
    sampled_queries = queries.sample(n=n, replace=True, random_state=int(rng.integers(1, 1_000_000))).reset_index(drop=True)
    dates = pd.Timestamp("2025-06-01") + pd.to_timedelta(rng.integers(0, 61, n), unit="D")
    hour_probs = np.array([0.035, 0.04, 0.055, 0.07, 0.08, 0.085, 0.075, 0.07, 0.065, 0.065, 0.075, 0.085, 0.08, 0.065, 0.055, 0.04])
    hour = rng.choice(range(8, 24), n, p=hour_probs / hour_probs.sum())
    return pd.DataFrame(
        {
            "session_id": [f"S{i:07d}" for i in range(1, n + 1)],
            "user_id": sampled_users["user_id"],
            "query_id": sampled_queries["query_id"],
            "session_date": dates.strftime("%Y-%m-%d"),
            "hour": hour,
            "device_type": rng.choice(["ios", "android", "mini_program"], n, p=[0.42, 0.48, 0.10]),
            "city_tier": sampled_users["city_tier"],
            "user_zone_x": sampled_users["home_zone_x"],
            "user_zone_y": sampled_users["home_zone_y"],
        }
    )


def _make_experiments(sessions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    variant = rng.choice(["holdout", "control", "treatment"], len(sessions), p=[0.15, 0.425, 0.425])
    out = sessions[["session_id", "city_tier"]].copy()
    out["experiment_id"] = "search_rank_2025q3"
    out["variant"] = variant
    out["assigned_at"] = sessions["session_date"]
    return out[["experiment_id", "session_id", "city_tier", "variant", "assigned_at"]]


def _make_impressions_and_orders(
    rng: np.random.Generator,
    users: pd.DataFrame,
    queries: pd.DataFrame,
    merchants: pd.DataFrame,
    sessions: pd.DataFrame,
    experiments: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users_i = users.set_index("user_id")
    queries_i = queries.set_index("query_id")
    exp_i = experiments.set_index("session_id")
    merchant_by_category = {cat: merchants.loc[merchants["merchant_category"] == cat].reset_index(drop=True) for cat in CATEGORIES}
    all_merchants = merchants.reset_index(drop=True)
    impressions: list[dict[str, object]] = []
    orders: list[dict[str, object]] = []

    for s in sessions.itertuples(index=False):
        user = users_i.loc[s.user_id]
        query = queries_i.loc[s.query_id]
        variant = exp_i.loc[s.session_id, "variant"]
        intent = str(query.intent_category)
        same_cat = merchant_by_category[intent].sample(n=min(k + 5, len(merchant_by_category[intent])), replace=False, random_state=int(rng.integers(1, 1_000_000)))
        others = all_merchants.loc[all_merchants["merchant_category"] != intent].sample(n=5, replace=False, random_state=int(rng.integers(1, 1_000_000)))
        cand = pd.concat([same_cat, others], ignore_index=True).sample(n=k, random_state=int(rng.integers(1, 1_000_000))).reset_index(drop=True)
        features = []
        for m in cand.itertuples(index=False):
            distance = _distance(float(s.user_zone_x), float(s.user_zone_y), float(m.zone_x), float(m.zone_y))
            category_match = int(m.merchant_category == intent)
            price_match = 1.0 - abs(_price_level_num(str(m.price_level)) - float(user.price_sensitivity))
            is_open = int(rng.random() < float(m.open_probability))
            expected_margin = float(m.avg_price) * (0.24 + 0.13 * float(m.service_quality))
            true_score = (
                1.35 * category_match
                + 0.60 * float(m.service_quality)
                + 0.40 * float(m.capacity_score)
                + 0.32 * (float(m.rating) - 3.0) / 2.0
                + 0.25 * (str(user.preferred_category) == intent)
                + 0.20 * is_open
                + 0.14 * float(query.query_urgency_score)
                - 0.42 * np.log1p(distance)
                + 0.18 * price_match
                + rng.normal(0, 0.12)
            )
            rule_score = (
                0.18 * category_match
                + 0.46 * (float(m.rating) - 3.0) / 2.0
                + 0.22 * is_open
                - 1.05 * np.log1p(distance)
                + 0.08 * price_match
                + rng.normal(0, 0.38)
            )
            modelish_score = 0.58 * true_score + 0.42 * rule_score + rng.normal(0, 0.08)
            features.append((m, distance, category_match, price_match, is_open, expected_margin, true_score, rule_score, modelish_score))
        if variant == "treatment":
            ordered = sorted(features, key=lambda x: x[8], reverse=True)
        elif variant == "control":
            ordered = sorted(features, key=lambda x: x[7] + rng.normal(0, 0.05), reverse=True)
        else:
            ordered = sorted(features, key=lambda x: x[7] + rng.normal(0, 0.25), reverse=True)

        session_has_order = False
        for rank, item in enumerate(ordered, start=1):
            m, distance, category_match, price_match, is_open, expected_margin, true_score, rule_score, modelish_score = item
            position_bias = 1.0 / np.log2(rank + 1.0)
            logged_propensity = max(0.04, position_bias * (0.82 if variant == "holdout" else 0.95))
            click_prob = _sigmoid(-2.55 + 1.10 * true_score) * position_bias
            click = int(rng.random() < click_prob)
            order_prob = _sigmoid(-3.50 + 1.18 * true_score + 0.35 * click) * position_bias
            order = int(click and not session_has_order and rng.random() < order_prob)
            revenue = 0.0
            gross_margin = 0.0
            if order:
                session_has_order = True
                revenue = max(12.0, float(query.avg_order_value) * rng.uniform(0.78, 1.35))
                gross_margin = revenue * (0.24 + 0.12 * float(m.service_quality))
                orders.append(
                    {
                        "order_id": f"O{len(orders) + 1:08d}",
                        "session_id": s.session_id,
                        "user_id": s.user_id,
                        "merchant_id": m.merchant_id,
                        "query_id": s.query_id,
                        "order_date": s.session_date,
                        "gross_revenue": round(revenue, 2),
                        "gross_margin": round(gross_margin, 2),
                    }
                )
            impressions.append(
                {
                    "impression_id": f"I{len(impressions) + 1:09d}",
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "query_id": s.query_id,
                    "merchant_id": m.merchant_id,
                    "variant": variant,
                    "logged_rank": rank,
                    "distance_km": round(distance, 4),
                    "rating": float(m.rating),
                    "price_match_score": round(float(price_match), 4),
                    "category_match": category_match,
                    "service_quality": float(m.service_quality),
                    "capacity_score": float(m.capacity_score),
                    "is_open": is_open,
                    "merchant_historical_ctr": float(m.merchant_historical_ctr),
                    "merchant_historical_cvr": float(m.merchant_historical_cvr),
                    "query_tail_score": float(query.query_tail_score),
                    "query_urgency_score": float(query.query_urgency_score),
                    "user_active_score": float(user.user_active_score),
                    "rank_position_signal": round(position_bias, 4),
                    "expected_margin": round(expected_margin, 2),
                    "rule_score": round(float(rule_score), 6),
                    "true_relevance_score": round(float(true_score), 6),
                    "logged_propensity": round(float(logged_propensity), 6),
                    "clicked": click,
                    "ordered": order,
                    "gross_revenue": round(float(revenue), 2),
                    "gross_margin": round(float(gross_margin), 2),
                }
            )
    return pd.DataFrame(impressions), pd.DataFrame(orders)


def _category_aov(category: str) -> float:
    return {
        "car_wash": 80.0,
        "tire_repair": 140.0,
        "chauffeur": 180.0,
        "maintenance": 420.0,
        "restaurant": 160.0,
        "parent_child": 220.0,
    }[category]


def _price_for_category(category: str, level: str, rng: np.random.Generator) -> float:
    base = _category_aov(category)
    multiplier = {"low": 0.72, "mid": 1.0, "high": 1.35}[level]
    return round(float(base * multiplier * rng.uniform(0.82, 1.20)), 2)


def _price_level_num(level: str) -> float:
    return {"low": 0.25, "mid": 0.55, "high": 0.88}[level]


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) * 0.65 + 0.2)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))
