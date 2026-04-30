# 面试讲稿：本地生活搜索排序 ML Lab

## 3 分钟版本

这个项目我想展示的是搜索排序和机器学习评估能力，而不是简单做一个预测模型。我模拟了一个本地生活 App 的搜索场景，用户搜索洗车、补胎、代驾、保养、餐饮、亲子等需求，系统要对候选商家排序，目标是提升点击、下单和 GMV，同时控制距离、供给覆盖和长尾 query 体验。

我生成了用户、query、商家、搜索 session、曝光、订单和实验分组等合成数据。模型部分有四层：规则 baseline、pointwise CTR/CVR 模型、pairwise ranker，以及融合 CVR、CTR、预期毛利、距离和供给能力的 reranker。

评估上我没有只看 AUC，而是用了 NDCG@5/10、MAP、MRR、CTR/CVR lift 和 IPS 反事实评估。因为搜索日志有位置偏差，历史点击不等于真实偏好，所以需要用 logged propensity 做校正。

最后报告里我还做了长尾 query、位置偏差、bad case 和上线 guardrail。这个项目能说明我理解搜索排序不只是模型训练，还包括业务指标、反事实评估、实验验证和策略落地。

## 8 分钟版本

讲述顺序：

1. 业务问题：本地生活搜索如何排序，为什么不能只看点击率。
2. 数据设计：用户、query、商家、session、impression、order、experiment。
3. 特征工程：query intent、tail type、距离、评分、价格匹配、品类匹配、营业、产能、历史 CTR/CVR。
4. 模型结构：rule baseline、pointwise、pairwise、reranker。
5. 评估体系：NDCG/MAP/MRR/AUC/Calibration/IPS。
6. 业务诊断：长尾 query、position bias、供给覆盖、bad case。
7. 上线建议：先小流量实验，监控 GMV、距离、低供给 query、商家集中度和用户体验。

面试官如果问为什么不用深度学习，可以回答：这个项目的重点是完整排序分析闭环。对于数据分析岗位，可解释的 sklearn 模型、反事实评估和业务 guardrail 比堆深度模型更能体现落地能力。
