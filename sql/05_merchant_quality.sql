SELECT
    m.merchant_category,
    COUNT(DISTINCT m.merchant_id) AS merchants,
    COUNT(i.impression_id) AS impressions,
    ROUND(AVG(m.rating), 2) AS avg_rating,
    ROUND(AVG(m.service_quality), 4) AS avg_service_quality,
    ROUND(AVG(i.clicked), 4) AS click_rate,
    ROUND(AVG(i.ordered), 4) AS order_rate,
    ROUND(SUM(i.gross_revenue), 2) AS gross_revenue,
    ROUND(SUM(i.gross_margin), 2) AS gross_margin
FROM merchants m
JOIN impressions i ON i.merchant_id = m.merchant_id
GROUP BY m.merchant_category
ORDER BY gross_margin DESC;
