SELECT
    logged_rank,
    COUNT(*) AS impressions,
    ROUND(AVG(clicked), 4) AS click_rate,
    ROUND(AVG(ordered), 4) AS order_rate,
    ROUND(AVG(logged_propensity), 4) AS avg_logged_propensity,
    ROUND(SUM(gross_revenue), 2) AS gross_revenue
FROM impressions
GROUP BY logged_rank
ORDER BY logged_rank;
