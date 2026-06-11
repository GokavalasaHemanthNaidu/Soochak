-- SOOCHAK Phase 5: Predictions table schema
-- Run this against your SQLite database to create the predictions table

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,

    -- Raw input features
    road_type TEXT NOT NULL,
    speed_limit INTEGER NOT NULL,
    weather TEXT NOT NULL,
    lighting TEXT NOT NULL,
    junction TEXT NOT NULL,
    junction_control TEXT NOT NULL,
    vehicle_type TEXT NOT NULL,
    driver_age TEXT NOT NULL,
    urban_rural TEXT NOT NULL,
    state TEXT NOT NULL,
    city TEXT NOT NULL,

    -- Prediction outputs
    predicted_class INTEGER NOT NULL,
    probability REAL NOT NULL,
    raw_probability REAL NOT NULL,
    confidence TEXT NOT NULL,
    inference_ms REAL NOT NULL,
    shap_base_value REAL NOT NULL,
    top_features_json TEXT NOT NULL,

    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Index for time-series queries (drift detection)
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
    ON predictions(timestamp);

-- Index for confidence analysis
CREATE INDEX IF NOT EXISTS idx_predictions_confidence 
    ON predictions(confidence);

-- Drift events table (Phase 8 unknown category drift monitoring)
CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    feature_name TEXT NOT NULL,
    value_raw TEXT,
    event_type TEXT DEFAULT 'unknown_category'
);

CREATE INDEX IF NOT EXISTS idx_drift_events_feature 
    ON drift_events(feature_name);
