WITH session_outcome AS (
    SELECT
        e.variant,
        s.session_id,
        MAX(i.clicked) AS clicked,
        MAX(i.ordered) AS ordered,
        SUM(i.gross_revenue) AS gross_revenue,
        SUM(i.gross_margin) AS gross_margin
    FROM search_sessions s
    JOIN experiments e ON e.session_id = s.session_id
    JOIN impressions i ON i.session_id = s.session_id
    GROUP BY e.variant, s.session_id
)
SELECT
    variant,
    COUNT(*) AS sessions,
    ROUND(AVG(clicked), 4) AS click_rate,
    ROUND(AVG(ordered), 4) AS order_rate,
    ROUND(SUM(gross_revenue), 2) AS gross_revenue,
    ROUND(SUM(gross_margin), 2) AS gross_margin,
    ROUND(SUM(gross_revenue) / COUNT(*), 2) AS gmv_per_session,
    ROUND(SUM(gross_margin) / COUNT(*), 2) AS margin_per_session
FROM session_outcome
GROUP BY variant
ORDER BY CASE variant WHEN 'holdout' THEN 1 WHEN 'control' THEN 2 WHEN 'treatment' THEN 3 ELSE 4 END;
