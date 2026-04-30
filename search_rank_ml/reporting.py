from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def write_report(results: dict[str, pd.DataFrame], metrics_path: Path, figures: dict[str, Path], report_path: Path) -> Path:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rel = {name: os.path.relpath(path, report_path.parent) for name, path in figures.items()}
    exp = results["experiment"].set_index("variant")
    treatment_lift = exp.loc["treatment", "order_rate"] - exp.loc["control", "order_rate"]
    ndcg_lift = metrics["ndcg10_lift_vs_rule"]
    lines = [
        "# 本地生活搜索排序、转化预估与反事实评估报告",
        "",
        "## 0. 项目说明",
        "",
        "本项目使用完全合成的本地生活搜索日志，模拟 query、候选商家、曝光位置、点击、下单、收入和实验分组。项目不包含线下真实主体、精确地理坐标、真实成交记录或个人信息。",
        "",
        "## 1. Executive Summary",
        "",
        f"- ML reranker 的 NDCG@10 相比规则 baseline 提升 {ndcg_lift:.1%}。",
        f"- treatment 实验组相对 control 的下单率提升 {treatment_lift:.2%}。",
        f"- pointwise 点击 AUC 为 {metrics['auc_click_pointwise']:.3f}，订单 AUC 为 {metrics['auc_order_pointwise']:.3f}。",
        f"- IPS 反事实评估下，rerank top3 GMV/session 为 {metrics['ips_top3']['rerank']['ips_gmv_per_session']:.2f}。",
        "",
        "## 2. 搜索转化漏斗",
        "",
        f"![搜索漏斗]({rel['funnel']})",
        "",
        _table(results["funnel"]),
        "",
        "## 3. 实验与真实业务 Lift",
        "",
        f"![实验 lift]({rel['experiment']})",
        "",
        _table(results["experiment"]),
        "",
        "## 4. 排序模型效果",
        "",
        f"![排序指标]({rel['ranking']})",
        "",
        _metrics_table(metrics["ranking"]),
        "",
        "这个项目把排序拆成规则 baseline、pointwise CTR/CVR、pairwise ranker 和多目标 reranker。面试讲述重点是：模型不是只追求 AUC，而是要看 NDCG、MRR、GMV、长尾 query 和反事实评估。",
        "",
        "## 5. IPS 反事实评估",
        "",
        f"![IPS]({rel['ips']})",
        "",
        _nested_table(metrics["ips_top3"]),
        "",
        "IPS 使用曝光位置 propensity 对历史日志做校正，用来回答“如果把新排序策略放到同一批流量上，预期点击、订单和 GMV 会怎样”。它仍然是离线估计，上线前需要小流量 A/B 验证。",
        "",
        "## 6. 长尾 Query 与位置偏差",
        "",
        f"![长尾表现]({rel['tail']})",
        "",
        _table(results["tail_performance"]),
        "",
        f"![位置偏差]({rel['position_bias']})",
        "",
        _table(results["position_bias"].head(10)),
        "",
        "## 7. 特征解释与 Bad Case",
        "",
        f"![特征重要性]({rel['features']})",
        "",
        f"![bad case]({rel['bad_cases']})",
        "",
        _table(results["bad_cases"].head(10)),
        "",
        "## 8. 上线 Guardrail",
        "",
        "- 不只看 overall NDCG，要按 query intent、tail type、城市层级、商家品类拆分。",
        "- 新策略必须监控零结果、服务距离、商家集中度、低供给 query 和退款风险。",
        "- 归因和 IPS 只是上线前判断，最终增量必须用 holdout/control/treatment 小流量实验确认。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_model_card(results: dict[str, pd.DataFrame], metrics_path: Path, model_card_path: Path) -> Path:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    lines = [
        "# 模型卡：本地生活搜索排序 Reranker",
        "",
        "## 模型目标",
        "",
        "在用户搜索本地生活服务时，对候选商家进行排序，提升点击、下单和 GMV，同时兼顾距离、供给能力和长尾 query 体验。",
        "",
        "## 模型结构",
        "",
        "- Rule baseline：距离、评分、价格匹配、营业状态。",
        "- Pointwise：预测点击与下单概率。",
        "- Pairwise：同一 session 内正负候选商家的差分学习。",
        "- Reranker：融合 CVR、CTR、pairwise score、预期毛利、距离和供给能力。",
        "",
        "## 关键指标",
        "",
        _metrics_table(metrics["ranking"]),
        "",
        f"- 点击 AUC：{metrics['auc_click_pointwise']:.3f}",
        f"- 订单 AUC：{metrics['auc_order_pointwise']:.3f}",
        f"- NDCG@10 lift vs rule：{metrics['ndcg10_lift_vs_rule']:.1%}",
        "",
        "## 使用边界",
        "",
        "本项目数据为合成数据，模型指标用于展示方法论。真实上线前需要接入真实日志、做特征稳定性检查、业务 guardrail 和小流量实验。",
        "",
        "## 风险与监控",
        "",
        "- Position bias 可能导致模型过度学习头部位置。",
        "- 高 GMV 商品可能挤压低价高满意度商家。",
        "- 长尾 query 需要单独监控 NDCG、供给覆盖和用户满意度。",
    ]
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.write_text("\n".join(lines), encoding="utf-8")
    return model_card_path


def _metrics_table(ranking: dict[str, dict[str, float]]) -> str:
    rows = []
    for model, vals in ranking.items():
        rows.append({"model": model, **vals})
    return _table(pd.DataFrame(rows))


def _nested_table(values: dict[str, dict[str, float]]) -> str:
    rows = []
    for model, vals in values.items():
        rows.append({"model": model, **vals})
    return _table(pd.DataFrame(rows))


def _table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_fmt)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep, *body])


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if 0 < abs(value) < 1:
            return f"{value:.2%}"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
