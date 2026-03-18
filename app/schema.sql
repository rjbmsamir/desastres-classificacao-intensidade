CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    finished_at TEXT,
    algorithm TEXT,
    params_json TEXT,
    metrics_summary_json TEXT,
    data_path TEXT,
    n_samples INTEGER,
    n_features INTEGER,
    target_positive TEXT DEFAULT 'raro'
);

CREATE TABLE IF NOT EXISTS cv_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    fold INTEGER,
    f1_macro REAL,
    balanced_accuracy REAL,
    precision_nivel_I REAL,
    recall_nivel_I REAL,
    f1_nivel_I REAL,
    precision_nivel_II REAL,
    recall_nivel_II REAL,
    f1_nivel_II REAL,
    precision_nivel_III REAL,
    recall_nivel_III REAL,
    f1_nivel_III REAL,
    precision_raro REAL,
    recall_raro REAL,
    f1_raro REAL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    source TEXT,
    row_hash TEXT,
    y_true TEXT,
    y_pred TEXT NOT NULL,
    proba_rara REAL,
    proba_comum REAL,
    probabilities_json TEXT,
    payload_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_unique
    ON predictions(run_id, source, row_hash);
