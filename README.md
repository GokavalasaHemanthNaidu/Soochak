<div align="center">

<img src="https://img.shields.io/badge/Explainable%20AI-Accident%20Severity%20Predictor-8A2BE2?style=for-the-badge&logo=polestar&logoColor=white" alt="Soochak"/>

# 🚨 SOOCHAK

### *Explainable Road Accident Severity Predictor + Intervention Simulator*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-000000?style=flat-square&logo=xgboost&logoColor=white)]()
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2?style=flat-square)]()
[![Status: Live](https://img.shields.io/badge/Status-Completed-9400D3?style=flat-square)]()

**[🚀 Live Demo](https://soochak.onrender.com)** · **[🐛 Report Bug](https://github.com/GokavalasaHemanthNaidu/Soochak/issues)**

</div>

---

SOOCHAK predicts fatal road accidents using real India FIR data, explains the predictions using SHAP, and simulates what-if interventions (e.g., adding a divider) to prevent them. Designed for a zero-cost Render deployment.

---

## 📉 Business Impact
- **The Problem**: India sees over 1.73 Lakh road crash deaths annually.
- **Economic Cost**: ₹5.96 Lakh crore (approx. 3.14% of GDP).
- **The Solution**: SOOCHAK provides targeted, explainable interventions to help reduce severe and fatal accidents at high-risk blackspots.

---

## 📊 Model Performance

Trained on Indian FIR road accident data. Evaluated on **Delhi geographic holdout** (unseen state).

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.72 |
| F1 Score (Fatal class) | 0.47 |
| Recall (Fatal) | 0.62 |
| Decision Threshold | 0.18 (F1-optimal) |
| Fatal Base Rate | 15.4% |
| Training States | 5 Indian states |
| Test State | Delhi (holdout) |

**Why F1 0.47?** Indian FIR data has 15–30% missing values and weak feature coverage. High recall (0.62) is intentional — missing a fatal is worse than a false alarm. The value of this project is in **SHAP explainability and what-if simulation**, not raw classification accuracy.

**Calibration:** Raw XGBoost outputs ~0.52 probability (overconfident). Isotonic calibration corrects this to ~0.20, matching the true 15.4% base rate. Threshold 0.18 captures the upper tail of risk.

---

## 🔒 Security & Quality (20/20 PASS)
- Strictly **NO SECRETS** committed (`.env` is fully blocked).
- 100% Pydantic validation on all incoming data.
- XSS sanitization (`html.escape`) on text fields.
- Automated CI pipeline with `pytest`, `ruff`, `bandit`, and `safety`.
- Rate limiting implemented (30 requests/min/IP).
- API Key gating for bot protection.

---

## ⚠️ Deployment Limitations (Render Free Tier)
This application uses **SQLite** in WAL mode for zero-infrastructure deployment.
*   **Ephemeral Storage**: Because it runs on Render's free tier, the SQLite database (`soochak.db`) is ephemeral. It resets when the container restarts.
*   **Upgrade Path**: For production, the `DB_PATH` and `aiosqlite` connection should be swapped out for a managed PostgreSQL database (e.g., Supabase, Neon) using Asyncpg.

---

## 🏗️ Architecture
[See Architecture Document](docs/architecture.md)

### ML Inference & Calibration
The isotonic calibrator corrects systematic overconfidence from the XGBoost model. Raw probabilities of ~0.60 are calibrated down to ~0.21, reflecting the true 15.4% base rate of the dataset. The optimal decision threshold of `0.18` maximizes F1 on the hold-out calibration set.

### Calibration Behavior

| Stage | Probability | Interpretation |
|-------|-------------|----------------|
| Raw XGBoost | ~0.52 | Overconfident — model thinks 52% fatal |
| Isotonic calibrated | ~0.20 | Corrected — matches true ~15% base rate |
| Decision threshold | 0.18 | Captures upper tail of risk distribution |

The raw-to-calibrated compression is expected and correct. It reflects the model learning spurious correlations from limited features and the calibrator correcting for this overconfidence using the hold-out set.

### Data Limitations & Target Encoding Pitfalls

The model uses target encoding for `State_Risk_Score` and `Weather_Risk`. Rare categories with few samples can produce unreliable encodings:

| Category | Samples | Target Encoding | Reliability |
|----------|---------|-----------------|-------------|
| "Normal" weather | High | 0.1603 | ✅ Reliable |
| "Raining" | Medium | 0.1288 | ✅ Reliable |
| "Windy" | Medium | 0.1639 | ✅ Reliable |
| "Fog or mist" | Very low | 0.0000 | ⚠️ Unreliable — zero fatal samples in training |

**Impact:** The model underestimates risk for fog because the training data happened to have no fatal fog accidents. This is a known limitation of target encoding on imbalanced categorical data, not a model bug. The threshold of 0.18 is calibrated against reliable categories.


## 📁 Project Structure
```text
soochak/
├── src/api/routers/
├── src/services/
├── src/db/migrations/
├── src/models/
├── src/middleware/
├── src/templates/
├── src/static/
├── src/utils/
├── tests/
├── ml_artifacts/
├── docs/
├── infra/
├── portfolio-hub/
└── .github/workflows/
```

---

## 🚀 Quick Start (Local Development)

1. **Clone & Environment Setup:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/soochak.git
   cd soochak
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Copy `.env.example` to `.env` and add your Groq API key:
   ```bash
   cp .env.example .env
   ```

4. **Run the API:**
   ```bash
   # Essential for Python path resolution
   export PYTHONPATH=.
   python -m src.api.main
   ```
   Visit `http://localhost:8000` for the dashboard.

---

## 📄 License
This project is licensed under the MIT License.
