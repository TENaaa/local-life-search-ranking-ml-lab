SELECT
    q.intent_category,
    q.tail_type,
    COUNT(DISTINCT s.session_id) AS sessions,
    ROUND(1.0 * COUNT(DISTINCT CASE WHEN i.ordered = 1 THEN s.session_id END) / COUNT(DISTINCT s.session_id), 4) AS order_rate,
    ROUND(AVG(i.category_match), 4) AS category_match_rate,
    ROUND(AVG(i.distance_km), 2) AS avg_distance_km,
    ROUND(AVG(i.capacity_score), 4) AS avg_capacity_score,
    ROUND(SUM(i.gross_revenue), 2) AS gross_revenue
FROM search_sessions s
JOIN queries q ON q.query_id = s.query_id
JOIN impressions i ON i.session_id = s.session_id
GROUP BY q.intent_category, q.tail_type
HAVING sessions >= 80
ORDER BY order_rate ASC, avg_distance_km DESC
LIMIT 20;
