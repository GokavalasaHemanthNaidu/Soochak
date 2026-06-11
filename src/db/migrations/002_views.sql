-- VIEW 1: Risk by Road Segment
CREATE VIEW IF NOT EXISTS v_risk_by_segment AS
SELECT
    road_type, speed_limit, lighting, junction,
    COUNT(*) as total_incidents,
    SUM(CASE WHEN severity_true = 1 THEN 1 ELSE 0 END) as fatal_count,
    ROUND(100.0 * SUM(CASE WHEN severity_true = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as fatal_rate_pct,
    AVG(p.probability) as avg_model_confidence
FROM incidents i
LEFT JOIN predictions p ON i.id = p.incident_id
GROUP BY road_type, speed_limit, lighting, junction
HAVING COUNT(*) >= 10;

-- VIEW 2: Intervention ROI
CREATE VIEW IF NOT EXISTS v_intervention_roi AS
WITH baseline AS (
    SELECT AVG(CASE WHEN severity_true = 1 THEN 1.0 ELSE 0 END) as base_fatal_rate
    FROM incidents
)
SELECT
    'Add divider on NH without divider' as intervention_name,
    COUNT(*) as affected_incidents,
    SUM(CASE WHEN severity_true = 1 THEN 1 ELSE 0 END) as current_fatal_count,
    ROUND((b.base_fatal_rate * COUNT(*) - SUM(CASE WHEN severity_true = 1 THEN 1 ELSE 0 END)) * 50000, 0) as est_savings_inr
FROM incidents i, baseline b
WHERE i.road_type = 'National Highway' AND i.junction_ctrl = 'Uncontrolled';

-- VIEW 3: Feature Drift Monthly
CREATE VIEW IF NOT EXISTS v_feature_drift_monthly AS
SELECT
    strftime('%Y-%m', datetime) as month,
    COUNT(*) as incident_volume,
    AVG(speed_limit) as avg_speed_limit,
    AVG(CASE WHEN lighting = 'Darkness' THEN 1.0 ELSE 0 END) as pct_darkness,
    AVG(CASE WHEN weather IN ('Rain','Fog') THEN 1.0 ELSE 0 END) as pct_adverse_weather
FROM incidents
GROUP BY strftime('%Y-%m', datetime)
ORDER BY month;

-- VIEW 4: Model Performance by Segment
CREATE VIEW IF NOT EXISTS v_model_performance AS
SELECT
    i.road_type, i.state,
    COUNT(*) as n_predictions,
    ROUND(AVG(CASE WHEN i.severity_true = 1 AND p.predicted_class = 1 THEN 1.0 ELSE 0 END), 3) as precision_fatal,
    ROUND(AVG(CASE WHEN i.severity_true = 1 AND p.predicted_class = 1 THEN 1.0 ELSE 0 END) /
          NULLIF(AVG(CASE WHEN i.severity_true = 1 THEN 1.0 ELSE 0 END), 0), 3) as recall_fatal,
    AVG(p.probability) as avg_confidence,
    AVG(ABS(p.probability - CASE WHEN i.severity_true = 1 THEN 1.0 ELSE 0.0 END)) as calibration_error
FROM incidents i
JOIN predictions p ON i.id = p.incident_id
WHERE i.severity_true IS NOT NULL
GROUP BY i.road_type, i.state;

-- VIEW 5: Feedback Summary
CREATE VIEW IF NOT EXISTS v_feedback_summary AS
SELECT
    p.model_version,
    COUNT(f.id) as total_feedback,
    SUM(CASE WHEN f.was_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
    ROUND(100.0 * SUM(CASE WHEN f.was_correct = 1 THEN 1 ELSE 0 END) / COUNT(f.id), 2) as accuracy_pct,
    SUM(CASE WHEN f.human_label = 1 AND p.predicted_class = 0 THEN 1 ELSE 0 END) as missed_fatal,
    SUM(CASE WHEN f.human_label = 0 AND p.predicted_class = 1 THEN 1 ELSE 0 END) as false_alarms
FROM predictions p
LEFT JOIN feedback f ON p.id = f.prediction_id
WHERE f.id IS NOT NULL
GROUP BY p.model_version;

-- VIEW 6: State-wise Risk Ranking
CREATE VIEW IF NOT EXISTS v_state_risk_ranking AS
SELECT
    state,
    COUNT(*) as total_incidents,
    SUM(CASE WHEN severity_true = 1 THEN 1 ELSE 0 END) as fatal_count,
    ROUND(100.0 * SUM(CASE WHEN severity_true = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as fatal_rate_pct,
    AVG(p.probability) as avg_model_confidence,
    RANK() OVER (ORDER BY SUM(CASE WHEN severity_true = 1 THEN 1 ELSE 0 END) DESC) as fatality_rank
FROM incidents i
LEFT JOIN predictions p ON i.id = p.incident_id
GROUP BY state
HAVING COUNT(*) >= 50;

-- VIEW 7: Latency & Cache Performance
CREATE VIEW IF NOT EXISTS v_latency_stats AS
SELECT
    endpoint,
    COUNT(*) as total_requests,
    ROUND(AVG(latency_ms), 2) as avg_ms,
    ROUND(MAX(latency_ms), 2) as max_ms,
    ROUND(AVG(CASE WHEN latency_ms < 50 THEN latency_ms END), 2) as p50_ms,
    SUM(rate_limited) as rate_limited_count,
    strftime('%Y-%m-%d', created_at) as date
FROM request_logs
GROUP BY endpoint, strftime('%Y-%m-%d', created_at);
