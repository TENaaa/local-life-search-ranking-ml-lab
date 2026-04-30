SELECT
    q.intent_category,
    q.tail_type,
    COUNT(DISTINCT s.session_id) AS sessions,
    COUNT(i.impression_id) AS impressions,
    ROUND(1.0 * SUM(CASE WHEN i.category_match = 1 THEN 1 ELSE 0 END) / COUNT(i.impression_id), 4) AS category_match_rate,
    ROUND(AVG(i.distance_km), 2) AS avg_distance_km,
    ROUND(AVG(i.is_open), 4) AS open_rate,
    ROUND(AVG(i.capacity_score), 4) AS avg_capacity_score
FROM search_sessions s
JOIN queries q ON q.query_id = s.query_id
JOIN impressions i ON i.session_id = s.session_id
GROUP BY q.intent_category, q.tail_type
ORDER BY sessions DESC;
