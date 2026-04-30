# 搜索排序指标体系

## 北极星指标

**搜索会话增量毛利**

口径：搜索 session 中由排序策略带来的订单毛利增量，最终需要用 holdout/control/treatment 实验确认。

## 排序质量指标

| 指标 | 说明 |
| --- | --- |
| NDCG@K | 关注高相关结果是否排在前面，适合多级相关性 |
| MAP@K | 关注前 K 个结果中的相关结果命中和排序 |
| MRR | 第一个相关结果出现得越早越好 |
| AUC | CTR/CVR 预估模型的区分能力 |
| Calibration | 预测概率是否接近真实发生率 |

## 业务指标

| 指标 | 说明 |
| --- | --- |
| CTR | 点击 session / 搜索 session |
| CVR | 下单 session / 搜索 session |
| GMV | 搜索订单收入 |
| 毛利 | GMV 扣除模拟履约成本后的利润 |
| GMV/session | 单次搜索会话收入 |
| IPS GMV/session | 用 propensity 校正后的反事实收入估计 |

## Guardrail

- 长尾 query 的 NDCG 和转化不能显著下降。
- 平均服务距离不能异常拉长。
- 低供给 query 的 category match rate 不能恶化。
- 商家集中度不能过高，避免头部商家挤压。
- 线上最终结论必须来自小流量 A/B 或 holdout 实验。
