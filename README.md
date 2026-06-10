# SOOCHAK: Explainable Road Accident Severity Predictor + Intervention Simulator

**Live Demo**: [https://soochak.onrender.com](https://soochak.onrender.com) *(Placeholder)*

SOOCHAK predicts fatal road accidents using real India FIR data, explains the predictions using SHAP, and simulates what-if interventions (e.g., adding a divider) to prevent them. Designed for a zero-cost Render deployment.

---

## 📉 Business Impact
- **The Problem**: India sees over 1.73 Lakh road crash deaths annually.
- **Economic Cost**: ₹5.96 Lakh crore (approx. 3.14% of GDP).
- **The Solution**: SOOCHAK provides targeted, explainable interventions to help reduce severe and fatal accidents at high-risk blackspots.

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
*   **Ephemeral Storage**: Because it runs on Render's free tier, the SQLite database (`roadrisk.db`) is ephemeral. It resets when the container restarts.
*   **Upgrade Path**: For production, the `DB_PATH` and `aiosqlite` connection should be swapped out for a managed PostgreSQL database (e.g., Supabase, Neon) using Asyncpg.

---

## 🏗️ Architecture
*Architecture diagram / ADRs placeholder*

## 📁 Project Structure
```text
roadrisk/
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
