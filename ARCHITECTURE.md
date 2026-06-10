# System Architecture

## Directory Structure
Describe how the folders are organized so the AI knows where to put new files.
```text
SOOCHAK/
├── venv/             # Python Virtual Environment
├── data/             # SQLite database directory
├── docs/             # Documentation and agent rules
├── infra/            # Dockerfile + render.yaml
├── ml_artifacts/     # ONNX model + calibrator + XGBoost
├── notebooks/        # Jupyter notebooks for ML training/eda
├── src/              # Main source code
│   ├── api/          # FastAPI main app and routers
│   ├── db/           # Database migrations and connection module
│   ├── middleware/   # Custom middlewares (rate limiting, etc.)
│   ├── models/       # Pydantic models (db models, request/response schemas)
│   ├── services/     # Business logic (predictions, etc.)
│   ├── static/       # CSS/JS for frontend
│   ├── templates/    # Jinja2 templates
│   └── utils/        # Helper utilities
└── tests/            # pytest suite
```

## Core Components
- **FastAPI Backend**: Serves predictions and hosts the Jinja2 dashboard.
- **ML Engine**: Uses ONNX runtime for low-latency XGBoost model inference.
- **Explainability Engine**: Uses SHAP for feature importance and Groq LLM API to generate human-readable explanations.
- **Async Database**: Uses `aiosqlite` with WAL mode for storing incidents, predictions, counterfactuals, and feedback.

## Data Flow
1. User submits an incident via the web dashboard (or API).
2. FastAPI processes the request, passes it to the ML Service.
3. ML Service runs ONNX inference and SHAP explainability.
4. Groq API translates SHAP values into a human explanation.
5. All data (incident, prediction, SHAP, Groq explanation) is saved to the SQLite database asynchronously.
6. The dashboard renders the result and counterfactuals (e.g., what if there was a divider).
