SELECT
    q.tail_type,
    COUNT(DISTINCT s.session_id) AS sessions,
    COUNT(i.impression_id) AS impressions,
    ROUND(AVG(i.clicked), 4) AS impression_ctr,
    ROUND(AVG(i.ordered), 4) AS impression_order_rate,
    ROUND(1.0 * COUNT(DISTINCT CASE WHEN i.clicked = 1 THEN s.session_id END) / COUNT(DISTINCT s.session_id), 4) AS session_click_rate,
    ROUND(1.0 * COUNT(DISTINCT CASE WHEN i.ordered = 1 THEN s.session_id END) / COUNT(DISTINCT s.session_id), 4) AS order_rate,
    ROUND(AVG(i.category_match), 4) AS category_match_rate,
    ROUND(AVG(i.distance_km), 2) AS avg_distance_km
FROM search_sessions s
JOIN queries q ON q.query_id = s.query_id
JOIN impressions i ON i.session_id = s.session_id
GROUP BY q.tail_type
ORDER BY CASE q.tail_type WHEN 'head' THEN 1 WHEN 'mid' THEN 2 ELSE 3 END;
