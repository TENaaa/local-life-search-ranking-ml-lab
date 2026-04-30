# 数据字典

所有数据均为合成数据。

## users

虚拟用户画像，包括城市层级、虚拟位置网格、价格敏感度、服务偏好、App 活跃和历史订单。

## queries

搜索 query，包括 query 文本、意图品类、head/mid/tail 类型、紧急程度和平均订单价值。

## merchants

虚拟商家，包括品类、城市层级、虚拟位置网格、评分、价格层级、服务质量、产能、营业概率、历史 CTR/CVR。

## search_sessions

搜索会话，包括用户、query、日期、小时、设备、城市层级和用户虚拟位置。

## experiments

实验分组，包括 `holdout`、`control`、`treatment`。

## impressions

曝光明细，一行代表一个 session 下的一个候选商家。核心字段包括排序位置、距离、类别匹配、价格匹配、质量、产能、propensity、点击、下单、收入和毛利。

## orders

订单表，只包含已下单 session 的订单收入和毛利。
