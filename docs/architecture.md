# System Architecture

## Architecture Diagram (ASCII)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOUD (Free Tier)                                   │
│                                                                             │
│   ┌──────────────┐        ┌─────────────────────────┐                      │
│   │ Google Colab │        │     Groq API            │                      │
│   │  (T4 GPU)    │        │  (llama-3-70b, free)    │                      │
│   │              │        │  1M tokens/day          │                      │
│   │ • EDA        │        │                         │                      │
│   │ • XGBoost    │        │ • SHAP → plain English  │                      │
│   │ • Optuna     │        │ • Policy recommendation │                      │
│   │ • SHAP       │        │                         │                      │
│   │ • ONNX Export│        └─────────────────────────┘                      │
│   └──────┬───────┘                  ▲                                      │
│          │ download (2MB .onnx)      │ async httpx (10s timeout, fallback)│
└──────────┼───────────────────────────┼────────────────────────────────────────┘
           │                           │
           ▼                           │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RENDER FREE TIER (Single Container)                    │
│                                                                             │
│  Browser / Postman                                                          │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │              FastAPI (uvicorn) :8000                 │                   │
│  │                                                      │                   │
│  │  ├── GET  /              ← HTML Dashboard (Jinja2)   │                   │
│  │  ├── GET  /docs          ← Swagger UI (auto)         │                   │
│  │  ├── POST /v1/predict    ← Prediction + Real SHAP    │                   │
│  │  ├── POST /v1/counterfactual ← What-if simulator     │                   │
│  │  ├── POST /v1/explain    ← Groq LLM + Fallback     │                   │
│  │  ├── POST /v1/batch/predict ← Async batch (Queue)    │                   │
│  │  ├── GET  /v1/batch/tasks/{id} ← Batch status        │                   │
│  │  ├── GET  /v1/incidents  ← Paginated history        │                   │
│  │  ├── GET  /v1/model/metrics ← Real registry metrics │                   │
│  │  ├── GET  /v1/model/drift   ← Encoded KS-test      │                   │
│  │  └── GET  /v1/health     ← DB + ONNX verification   │                   │
│  │                                                      │                   │
│  │  SQLite (WAL mode) ← ./data/soochak.db (ephemeral)  │                   │
│  │  ONNX Runtime ← 2MB model in memory                  │                   │
│  │  XGBoost Model ← TreeExplainer for real SHAP         │                   │
│  │  In-memory cache ← SHA256 dict + TTL 300s           │                   │
│  │  asyncio.Queue ← Lifespan background consumer      │                   │
│  │                                                      │                   │
│  └──────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## System Data Flow (Mermaid)

```mermaid
graph TB
    subgraph Training["Training (Google Colab Free T4)"]
        A[Kaggle Dataset] --> B[Polars EDA]
        B --> C[Data Cleaning]
        C --> D[XGBoost + Optuna]
        D --> E[Real SHAP TreeExplainer]
        E --> F[ONNX Export + Parity Check]
        F --> G[Isotonic Calibration]
        G --> H[Save XGB for SHAP]
    end

    subgraph Serving["Serving (Render Free Tier)"]
        I[FastAPI + Uvicorn] --> J[Jinja2 Dashboard]
        I --> K[ONNX Runtime]
        I --> L[Real SHAP Service]
        I --> M[Groq API Client + Fallback]
        I --> N[aiosqlite WAL]
        I --> O[Rate Limiter + API Key Gate]
        I --> P[asyncio.Queue]
    end

    H -->|500KB .pkl| L
    G -->|2MB .onnx| K
    G -->|50KB .pkl| K
    M -->|httpx async + fallback| Q[Groq API]
```
