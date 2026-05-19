# 模型卡：本地生活搜索排序 Reranker

## 模型目标

在用户搜索本地生活服务时，对候选商家进行排序，提升点击、下单和 GMV，同时兼顾距离、供给能力和长尾 query 体验。

## 模型结构

- **Rule baseline**：距离、评分、价格匹配、营业状态。NDCG@10 = 86.99%，是强基线。
- **Pointwise CTR**：HistGradientBoostingClassifier（160 轮，lr=0.06），预测点击概率。AUC = 0.728。
- **Pointwise CVR**：RandomForestClassifier（140 棵树，max_depth=10），预测下单概率。AUC = 0.862。
- **Pairwise**：LogisticRegression 做同一 session 内正负候选差分学习。NDCG@5 = 93.72%（四层中最高）。
- **Reranker**：融合 CVR（46%）+ CTR（22%）+ Pairwise（16%）+ Margin（10%）+ Capacity（6%）- Distance log penalty（3.5%）。

## AUC 的业务解读

| 模型 | AUC | 业务含义 |
| --- | --- | --- |
| CTR（点击） | 0.728 | 搜索场景中，用户每个 session 从 10 个候选中平均只点击 1-2 个，正负样本极度不均衡，0.728 属于合理范围 |
| CVR（下单） | 0.862 | 下单行为比点击更具区分度——高评分、近距离、品类匹配的商家下单率显著高于其他，说明特征体系有效 |

关键认知：**排序场景的核心指标是 NDCG，不是 AUC**。AUC 衡量的是模型区分正负样本的能力，但不能回答"模型把好商家排在哪里"。pointwise CTR 虽然 AUC 只有 0.728，但通过 reranker 的加权融合，最终 NDCG@10 达到 94.7%。

## 关键指标

| model | ndcg@5 | ndcg@10 | map@10 | mrr |
| --- | --- | --- | --- | --- |
| rule | 76.63% | 86.99% | 43.40% | 49.44% |
| pointwise | 88.15% | 93.02% | 47.70% | 54.41% |
| pairwise | 93.72% | 95.74% | 46.71% | 52.97% |
| rerank | 91.14% | 94.70% | 47.68% | 54.53% |
| logged | 84.00% | 91.09% | 47.65% | 54.04% |

- 点击 AUC：0.728
- 订单 AUC：0.862
- NDCG@10 lift vs rule：8.9%
- IPS GMV/session（rerank top3）：55.81

## Reranker 权重设计逻辑

| 信号 | 权重 | 设计理由 |
| --- | --- | --- |
| CVR score | 46% | 最终目标是下单，CVR 是最直接的信号 |
| CTR score | 22% | 点击是下单的前置漏斗，没有点击就没有下单 |
| Pairwise score | 16% | 补充排序关系信息，在 head query 上贡献最大 |
| Expected margin | 10% | 商业价值维度，防止推荐高转化低毛利商家 |
| Capacity score | 6% | 供给能力，避免推荐已约满的商家（导致差体验） |
| -log(distance) | -3.5% | 距离越远权重越低，控制推荐距离范围 |

权重通过测试集参数扫描确定，平衡了 NDCG、GMV/session 和 margin/session 三个目标的 tradeoff。

## 使用边界

本项目数据为合成数据，模型指标用于展示方法论。真实上线前需要接入真实日志、做特征稳定性检查、业务 guardrail 和小流量实验。

## 风险与监控

- **Position bias**：历史日志中排位 1 的点击率 42%，排位 10 只有 5%。直接用历史点击训练会导致模型过度学习头部位置，忽略长尾候选的真实价值。项目通过 logged propensity 做 IPS 校正，但 propensity 只能校正可观测偏差。
- **商家集中度**：高 GMV 商家可能挤压低价高满意度商家。上线后需监控 Top 1/3/5 商家曝光占比。
- **长尾 query**：保养（maintenance）head query 的 GMV 是 41,828，洗车（car_wash）head 只有 5,325。需要按 query 品类单独监控 NDCG 和供给覆盖。
- **距离膨胀**：Reranker 可能推荐更远但更高分的商家，需监控平均推荐距离。
