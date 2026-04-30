WITH flags AS (
    SELECT
        s.session_id,
        MAX(CASE WHEN i.logged_rank IS NOT NULL THEN 1 ELSE 0 END) AS has_results,
        MAX(i.clicked) AS clicked,
        MAX(i.ordered) AS ordered,
        SUM(i.gross_revenue) AS gross_revenue
    FROM search_sessions s
    LEFT JOIN impressions i ON i.session_id = s.session_id
    GROUP BY s.session_id
),
stages AS (
    SELECT '01_search_sessions' AS stage, COUNT(*) AS users FROM flags
    UNION ALL SELECT '02_has_results', SUM(has_results) FROM flags
    UNION ALL SELECT '03_clicked', SUM(clicked) FROM flags
    UNION ALL SELECT '04_ordered', SUM(ordered) FROM flags
)
SELECT
    stage,
    users,
    ROUND(1.0 * users / MAX(CASE WHEN stage = '01_search_sessions' THEN users END) OVER (), 4) AS rate_of_sessions,
    ROUND(1.0 * users / LAG(users) OVER (ORDER BY stage), 4) AS step_rate
FROM stages
ORDER BY stage;
