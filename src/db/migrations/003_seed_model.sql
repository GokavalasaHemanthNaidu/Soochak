-- Seed active model on first run if table is empty
INSERT OR IGNORE INTO model_registry 
(version, f1_score, roc_auc, precision_fatal, recall_fatal, onnx_path, calibrator_path, training_rows, is_active)
VALUES 
('v1.0.0', 0.65, 0.82, 0.68, 0.62, './ml_artifacts/soochak_v1.onnx', './ml_artifacts/calibrator.pkl', 15000, 1);
