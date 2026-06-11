import os

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "models", "response.py"))

append_str = """
class CounterfactualResponse(BaseModel):
    original_prob: float
    new_prob: float
    delta_prob: float
    delta_pct: float
    interpretation: str

class ExplainResponse(BaseModel):
    groq_explanation: str
    top_features: List[SHAPContribution]
    recommendation: Optional[str] = None

class BatchStatusResponse(BaseModel):
    task_id: str
    status: str
    progress_pct: float
    results_url: Optional[str] = None
    estimated_seconds: Optional[int] = None
    warning: str = "Jobs are stored in-memory and lost on container restart."

class IncidentResponse(BaseModel):
    pass

class IncidentListResponse(BaseModel):
    incidents: List[dict]
    total: int
    page: int
    pages: int

class ModelMetricsResponse(BaseModel):
    model_version: str
    f1_score: float
    roc_auc: float
    precision_fatal: float
    recall_fatal: float
    predictions_today: int
    cache_hit_rate: float

class DriftFeature(BaseModel):
    feature: str
    p_value: float
    drifted: bool

class DriftResponse(BaseModel):
    overall_drift: bool
    features: List[DriftFeature]
    checked_at: str

class FeedbackResponse(BaseModel):
    feedback_id: int
    correction_rate_7d: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
    db_connected: bool
    model_loaded: bool
    uptime_s: float

class LatencyResponse(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    last_1000_requests: int
"""

with open(file_path, "a") as f:
    f.write(append_str)

print("Appended missing models successfully.")
