# 本地生活搜索排序、转化预估与反事实评估

这是一个面向数据分析 / 搜索推荐 / 增长策略岗位的机器学习作品集项目。项目使用 100% 合成数据，模拟本地生活 App 的搜索场景：用户搜索洗车、补胎、代驾、保养、餐饮、亲子等需求，系统对候选商家排序，并通过离线排序指标、反事实评估和实验分析判断策略是否值得上线。

项目不包含线下真实主体、精确地理坐标、真实成交记录、真实品牌或个人信息。

## 项目亮点

- 搜索排序完整链路：query、候选召回、曝光位置、点击、下单、GMV、实验分组。
- 复杂 ML 展示：规则 baseline、pointwise CTR/CVR、pairwise ranker、多目标 reranker。
- 评估体系：NDCG@5/10、MAP@10、MRR、AUC、Calibration、GMV lift、IPS 反事实评估。
- 业务诊断：长尾 query、位置偏差、供给覆盖、bad case query、上线 guardrail。
- 中文报告、模型卡、SQL、图表和测试都可一键生成。

## 如何运行

```bash
python3 -m pip install -r requirements.txt
python3 -m search_rank_ml run-all
```

分步运行：

```bash
python3 -m search_rank_ml generate
python3 -m search_rank_ml build-db
python3 -m search_rank_ml train
python3 -m search_rank_ml evaluate
python3 -m search_rank_ml report
```

验证：

```bash
PYTHONPYCACHEPREFIX=/tmp/search_rank_pycache python3 -m compileall .
python3 tests/smoke_test.py
```

## 关键产物

- `reports/search_ranking_ml_case.md`：中文主报告
- `reports/model_card.md`：模型卡
- `reports/figures/`：漏斗、实验、排序指标、IPS、长尾 query、位置偏差、特征、bad case
- `output/ranking_metrics.json`：模型与反事实评估指标
- `output/models/ranking_predictions.csv`：测试集排序结果
- `sql/`：8 个可复用 SQL 分析主题
- `docs/metric_framework.md`：指标体系
- `docs/data_dictionary.md`：数据字典

## 面试讲述重点

这个项目不是只做一个分类模型，而是模拟了搜索排序从数据、模型、评估到上线 guardrail 的闭环。重点可以讲三件事：

1. 为什么排序不能只看 AUC，需要看 NDCG、MRR、GMV 和分 query 表现。
2. 为什么历史日志有 position bias，所以要用 propensity / IPS 做反事实评估。
3. 为什么 overall lift 不够，还要看长尾 query、供给覆盖、距离、商家集中度和实验 guardrail。

## License

MIT
