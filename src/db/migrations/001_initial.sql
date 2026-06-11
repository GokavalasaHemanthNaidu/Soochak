CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime TEXT,
    road_type TEXT,
    speed_limit INTEGER,
    weather TEXT,
    lighting TEXT,
    junction TEXT,
    junction_ctrl TEXT,
    vehicle_type TEXT,
    driver_age TEXT,
    urban_rural TEXT,
    state TEXT,
    city TEXT,
    lat REAL,
    lng REAL,
    severity_true INTEGER,
    source TEXT DEFAULT 'api',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    model_version TEXT,
    predicted_class INTEGER,
    probability REAL,
    shap_json TEXT,
    shap_base_value REAL,
    groq_explanation TEXT,
    inference_ms REAL,
    cache_hit INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS counterfactuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    changed_feature TEXT,
    original_value TEXT,
    new_value TEXT,
    original_prob REAL,
    new_prob REAL,
    delta_prob REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    was_correct INTEGER,
    human_label INTEGER,
    feedback_source TEXT DEFAULT 'ui',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT UNIQUE,
    f1_score REAL,
    roc_auc REAL,
    precision_fatal REAL,
    recall_fatal REAL,
    onnx_path TEXT,
    calibrator_path TEXT,
    training_rows INTEGER,
    is_active INTEGER DEFAULT 0,
    deployed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_ip_hash TEXT,
    endpoint TEXT,
    method TEXT,
    status_code INTEGER,
    latency_ms REAL,
    rate_limited INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    feature_name TEXT NOT NULL,
    value_raw TEXT,
    event_type TEXT DEFAULT 'unknown_category'
);

CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_incident ON predictions(incident_id);
CREATE INDEX IF NOT EXISTS idx_incidents_datetime ON incidents(datetime);
CREATE INDEX IF NOT EXISTS idx_incidents_state ON incidents(state);
CREATE INDEX IF NOT EXISTS idx_incidents_city ON incidents(city);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON request_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_request_logs_ip ON request_logs(client_ip_hash);
CREATE INDEX IF NOT EXISTS idx_drift_events_feature ON drift_events(feature_name);

