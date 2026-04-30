SELECT
    COUNT(*) AS rows,
    ROUND(AVG(distance_km), 2) AS avg_distance_km,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(category_match), 4) AS category_match_rate,
    ROUND(AVG(is_open), 4) AS open_rate,
    ROUND(AVG(clicked), 4) AS click_rate,
    ROUND(AVG(ordered), 4) AS order_rate,
    ROUND(AVG(gross_revenue), 2) AS avg_revenue_per_impression
FROM impressions;
