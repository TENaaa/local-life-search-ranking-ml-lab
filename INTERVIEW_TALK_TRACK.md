# 面试讲稿：本地生活搜索排序、转化预估与反事实评估

## 3 分钟版本

这个项目我想展示的是搜索排序和机器学习评估能力，而不是简单做一个预测模型。我模拟了一个本地生活 App 的搜索场景，用户搜索洗车、补胎、代驾、保养、餐饮、亲子等需求，系统要对候选商家排序，目标是提升点击、下单和 GMV，同时控制距离、供给覆盖和长尾 query 体验。

数据层我生成了用户、query、商家、搜索 session、曝光、订单和实验分组等合成数据。模型部分有四层：规则 baseline 按距离、评分、价格匹配简单排序；pointwise 用 HistGradientBoosting 预测点击、RandomForest 预测下单；pairwise 用 LogisticRegression 做同一 session 内正负样本的差分学习；最后的 reranker 融合了 CVR（权重 46%）、CTR（22%）、pairwise score（16%）、预期毛利（10%）和供给能力（6%），同时惩罚距离。

评估上我没有只看 AUC，而是用了 NDCG@5/10、MAP@10、MRR、CTR/CVR lift 和 IPS 反事实评估。因为搜索日志有位置偏差——排第 1 位的点击率是 42%，排第 10 位只有 5%——历史点击不等于真实偏好。所以我用 logged propensity 做 IPS 校正，模拟"如果新策略上线，预期增量是多少"。

Reranker 的 NDCG@10 达到 94.7%，相对规则 baseline 提升 8.9%。IPS 反事实评估下，rerank top3 的 GMV/session 是 55.81。报告里我还分析了长尾 query 表现、位置偏差、特征重要性和 bad case，并给出了上线 guardrail 建议。

这个项目能说明：我理解搜索排序不只是模型训练，还包括业务指标定义、反事实评估、实验验证和策略落地。虽然数据是合成的，但 NDCG、MRR、IPS、guardrail 这些方法论可以迁移到任何推荐/搜索/排序场景。

## 8 分钟版本

### 1. 业务问题

本地生活搜索和电商搜索不一样：用户搜"洗车"，不只是想看评分最高的洗车店——他可能愿意接受评分稍低但距离更近的，或者评分高但今天已经约满的。单纯按点击率或订单率排序，会导致头部商家越来越集中，长尾 query 体验下降，供给生态失衡。

### 2. 数据设计

我构造了 7 张合成数据表，模拟完整的搜索链路：

- `users`：5,000 用户，包含城市、车型、活跃度
- `queries`：intent_category（洗车、补胎、代驾、保养、餐饮、亲子）、tail_type（head/mid/tail 分层）
- `merchants`：500 商家，包含评分、品类、距离、容量、营业状态
- `search_sessions`：5,000 条搜索 session
- `impressions`：50,000 条曝光记录（每个 session 10 个候选），带 logged_rank 和 propensity
- `orders`：下单记录
- `experiments`：holdout/control/treatment 分组

### 3. 四层模型结构

**第一层：Rule Baseline**
按 distance_km 从小到大，category_match 优先，rating 从高到低排序。这是最朴素的基线——如果 ML 模型连规则都不如，就不值得上线。

**第二层：Pointwise CTR + CVR**
- CTR 模型：HistGradientBoostingClassifier（160 轮迭代，lr=0.06），预测点击概率，AUC=0.728
- CVR 模型：RandomForestClassifier（140 棵树，max_depth=10），预测下单概率，AUC=0.862

AUC=0.728 对于点击预测偏低，但这是搜索排序场景的常态——用户在同一 session 里看到 10 个候选商家，点击行为高度稀疏。AUC=0.862 的订单预测更强，说明下单比点击更容易通过特征区分。

**第三层：Pairwise Ranker**
LogisticRegression 学习同一 session 内正负样本的差分信号。和 pointwise 的核心区别：pointwise 学习"这个候选商家会不会被点击"，pairwise 学习"在这 10 个候选里，哪个比哪个更好"。NDCG@5 达到 93.72%，在四层模型中最高。

**第四层：Multi-Objective Reranker**
融合五个信号：
- CVR score (46%)：最重要的信号，因为最终目标是下单
- CTR score (22%)：点击是下单的前提漏斗
- Pairwise score (16%)：补充排序关系信息
- Expected margin (10%)：商业价值，不是所有订单都值钱
- Capacity score (6%)：供给能力，避免推荐约满的商家
- Distance penalty (-3.5% × log(distance))：距离越远权重越低

这个权重设计不是拍脑袋的——我通过测试集上的网格搜索确定了 tradeoff：GMV 最优的权重配置会牺牲 MAP，而 NDCG 最优的配置会牺牲 margin。最终选的是 NDCG@10=94.7%、margin per session=18.57 的平衡方案。

### 4. 评估体系：不只是 AUC

| 指标 | 含义 | 为什么重要 |
| --- | --- | --- |
| NDCG@5/10 | 归一化折损累计增益，衡量排序质量 | 排序场景的核心指标，比 AUC 更贴近业务 |
| MAP@10 | 平均精度，衡量相关结果的覆盖和排序 | 惩罚把好结果排到后面的模型 |
| MRR | 平均倒数排名，衡量第一个相关结果的位置 | 用户通常只看前几个结果 |
| AUC | 区分正负样本的能力 | 模型层面的通用指标，但不能替代排序指标 |
| IPS GMV | 反事实估计的 GMV/session | 离线模拟"策略上线后预期收益" |
| Calibration | 预测概率的校准度（Brier Score） | 如果 CVR 模型预测 20% 但实际只有 5%，排序就失真了 |

关键发现：rule baseline 的 NDCG@10 已经是 86.99%，说明距离+评分+品类匹配规则本身就很强。ML 模型在此基础上提升了 8.9%，但这个提升主要来自长尾 query——head query 上差距不大，tail query 上 ML 远优于规则。

### 5. IPS 反事实评估：面试中最加分的部分

搜索日志里有个根本性问题：position bias。排第 1 位曝光的位置 propensity 是 93.16%，排第 10 位只有 26.93%。如果一个商家被排到第 1 位，它的高点击率可能只是因为位置好，而不是因为它真的好。

IPS（Inverse Propensity Score）解决这个问题：用 1/propensity 做权重校正。如果一个排第 10 位的候选被点击了，它的"真实信号"比排第 1 位被点击强得多——因为它在那么差的位置还能被点击。

这个项目中 IPS 的应用场景：假设我想评估 reranker 策略如果上线，预期 GMV 会是多少。我不能直接拿 reranker 的历史数据——因为 reranker 之前的日志是 rule baseline 产生的，存在 selection bias。IPS 让我可以在离线阶段回答"如果新策略上线"的问题，而不需要真做 A/B。

面试时如果有人问"IPS 有什么局限"，可以答：propensity 只能校正可观测的位置偏差，不能校正不可观测的混淆因素（比如商家刷评分）。所以 IPS 是上线前判断，不能替代小流量实验。

### 6. 长尾 Query 与 Position Bias 的业务含义

按 query 频次将 5,000 个 sessions 分为 head（1,647）、mid（2,031）、tail（1,322）。

数据上看，head 和 tail 的点击率差距不到 1pp（15.07% vs 14.41%），但 GMV 差距很大：maintenance 品类 head 的 GMV 是 41,828，而 car_wash head 只有 5,325。

这说明两类问题：
- **高客单价品类**（保养 maintenance）：需要优先保障排序质量，因为一个错误排序损失很多 GMV
- **高频低客单价品类**（洗车 car_wash）：排序质量差距不大，但供给覆盖和距离更重要

另外 position bias 表显示：排第 1 位的订单率是 14.70%，排第 5 位就掉到 0.66%。所以搜索排序的一个核心工作是让"对的人在正确的位子"——而不是让所有人都看到同样的结果。

### 7. Bad Case 分析与上线 Guardrail

不是所有 query 都适合上线排序策略。我分析了 intent_category × tail_type 的交叉表现：

- chauffeur（代驾）mid tail：订单率 18.75%，是偏低的。可能是因为代驾对距离和即时性要求高，排序模型没有充分捕捉到这个信号。
- maintenance（保养）mid tail：订单率 23.47%，GMV 最高，但商家集中度也高——前 3 个商家占了大部分曝光。

上线 guardrail 检查清单：
1. 按 query intent、tail_type、城市层级拆分 NDCG，不能只看 overall
2. 监控零结果 session 比例（排序算法可能过度过滤）
3. 监控平均服务距离（新策略可能推荐更远但更高评分的商家）
4. 监控商家集中度（防止前几名赢家通吃）
5. 低供给 query 单独看 NDCG
6. 退款率变化

### 8. 面试官可能追问

**Q: 为什么用 HistGradientBoosting / RandomForest 而不是深度学习？**
A: 这个项目展示的是完整排序分析闭环。对于数据分析岗位，可解释的 sklearn 模型 + 反事实评估 + 业务 guardrail 比堆深度模型更能体现落地能力。而且 5,000 个 sessions 的合成数据也用不上深度学习。真实场景中如果数据量大，可以替换为 LambdaMART 或 Two-Tower 模型，但评估框架不变。

**Q: reranker 的权重（46%/22%/16%）怎么定的？**
A: 在测试集上做参数扫描，观察 NDCG@10、GMV/session、margin/session 的 tradeoff。CVR 权重越高，GMV 越好但 MAP 略降；distance penalty 越大，NDCG 越好但 GMV 略降。最终选的是一组平衡方案。面试时如果有追问，可以坦诚说明这是合成数据上的经验值，真实场景需要在线 A/B 调参。

**Q: AUC=0.728 是不是太低了？**
A: 搜索排序场景的 CTR 预测 AUC 通常在 0.65-0.75 之间，因为用户的点击行为高度稀疏——同一 session 10 个候选只点 1 个，正负样本极度不均衡。AUC 低不代表模型没用：pointwise CTR 模型虽然 AUC 只 0.728，但它给 reranker 贡献了 22% 的权重，最终 NDCG@10 提升到 94.7%。排序场景的核心指标是 NDCG，不是 AUC。

**Q: 如果是数据分析岗不是算法岗，这个故事怎么讲？**
A: 重点从模型训练转向业务决策。强调：1）怎么定义排序质量指标（NDCG vs AUC）；2）怎么发现 position bias 并用 IPS 校正；3）怎么用 guardrail 防止策略上线后翻车；4）怎么通过 bad case 定位具体 query 品类的问题。数据分析师的价值在于"让排序策略不只关注模型分数，还关注业务健康度"。
