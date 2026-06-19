"""Streamlit portfolio hub — multi-role selector for recruiters."""

import streamlit as st

st.set_page_config(page_title="SOOCHAK Portfolio Hub", layout="wide")

st.title("🚗 SOOCHAK Portfolio Hub")
st.markdown("**One project. Multiple angles. Pick your role.**")
st.markdown("---")

roles = {
    "SDE": {
        "emphasis": "Async REST API (/what-if <50ms), BackgroundTasks, SQLite WAL, Docker, CI/CD, XSS sanitization, API key gate, rate limiting",
        "link": "https://github.com/GokavalasaHemanthNaidu/Soochak/tree/main/src/api",
        "line": "Built an async FastAPI system with sub-50ms what-if simulations, real SHAP TreeExplainer, and SQLite WAL mode — deployed on Render with zero infrastructure cost.",
        "q": "Why asyncio.Queue over Celery? — No Redis needed on free tier; same producer-consumer pattern, in-process."
    },
    "Data Scientist": {
        "emphasis": "XGBoost + Optuna HPO, geographic cross-validation (5 states → Delhi holdout), real SHAP TreeExplainer, isotonic calibration, F1 0.47 / AUC 0.72",
        "link": "https://github.com/GokavalasaHemanthNaidu/Soochak/tree/main/notebooks",
        "line": "Trained XGBoost on messy Indian FIR data with geographic CV and isotonic calibration — F1 0.47 on Delhi holdout, AUC 0.72, threshold 0.18 (F1-optimal).",
        "q": "Why isotonic over Platt? — Platt assumes sigmoid-shaped miscalibration; isotonic is non-parametric and works better when raw probs are far from true base rate (52% → 15.4%)."
    },
    "MLE": {
        "emphasis": "ONNX export + 1e-5 parity validation, model registry, drift KS-test endpoint, health endpoint with ONNX verification, versioned artifacts in GitHub Releases",
        "link": "https://github.com/GokavalasaHemanthNaidu/Soochak/tree/main/ml_artifacts",
        "line": "Exported XGBoost to ONNX with 1e-5 parity validation, built a model registry with drift KS-test monitoring, deployed on Render with zero infra.",
        "q": "What is the KS-test monitoring? — Kolmogorov-Smirnov test compares live prediction distribution vs training distribution; large KS stat signals distribution shift."
    },
    "Applied Scientist": {
        "emphasis": "Real SHAP intervention simulator (what-if counterfactual), calibrated probabilities, Groq LLM explanation with rule-based fallback, fog encoding limitation documented",
        "link": "https://soochak.onrender.com",
        "line": "Built a real SHAP-powered intervention simulator showing adding a road divider reduces fatal probability by ~45% — with resilient LLM explanations and documented fog encoding limitation.",
        "q": "What does a SHAP value of +0.3 mean? — The feature pushed the log-odds of a fatal prediction up by 0.3 relative to the baseline expected prediction."
    },
    "Data Engineer": {
        "emphasis": "Polars ETL pipeline, 15-30% missing value imputation, messy vehicle type standardization, Parquet intermediate, SQLite star schema",
        "link": "https://github.com/GokavalasaHemanthNaidu/Soochak/tree/main/src/utils",
        "line": "Engineered a Polars ETL pipeline that imputes 15-30% missing values and standardizes messy free-text vehicle types into a clean star schema for XGBoost.",
        "q": "Why Polars over Pandas? — Lazy evaluation, Rust backend, 3-5x faster on multi-column transforms; no index to manage."
    },
    "MLOps Engineer": {
        "emphasis": "GitHub Actions CI (pytest + ruff + Bandit SAST + Safety), Docker multi-stage build, non-root container user, model registry, drift monitoring endpoint, branch protection docs",
        "link": "https://github.com/GokavalasaHemanthNaidu/Soochak/tree/main/.github/workflows",
        "line": "Set up full CI/CD with ruff linting, Bandit SAST, Safety dependency audit, pytest coverage — plus Docker multi-stage build with non-root user and model versioning.",
        "q": "What does Bandit check? — Static analysis for common Python security issues: hardcoded secrets, SQL injection risks, use of assert in production, weak crypto."
    },
    "Analytics / BI": {
        "emphasis": "Business impact framing (1.73L deaths/yr, ₹5.96L crore GDP cost), SHAP as policy intervention tool, what-if simulator for infrastructure decisions, paginated incident history",
        "link": "https://soochak.onrender.com",
        "line": "Framed a road safety ML model as a policy intervention tool — SHAP values translate directly into infrastructure decisions (add divider, reduce speed limit) with quantified risk reduction.",
        "q": "How would a traffic authority use this? — Input a high-risk corridor's features, run what-if: change road type from undivided to divided, see probability drop, prioritize budget accordingly."
    },
    "EDA / Notebook Analyst": {
        "emphasis": "EDA narrative: missing value audit, class imbalance (15.4% fatal), state-wise risk distribution, feature correlation heatmap, target encoding reliability table",
        "link": "https://github.com/GokavalasaHemanthNaidu/Soochak/tree/main/notebooks",
        "line": "Full EDA on messy Indian FIR data: identified 15-30% missing per column, discovered fog encoding failure (0 fatal fog samples in training), built state-risk scores, validated geographic CV split.",
        "q": "What was your most surprising EDA finding? — Fog or mist had zero fatal accidents in training states due to data sparsity, causing the model to assign 0% fatal risk to fog — documented as known limitation with fix strategy."
    },
    "Consulting / Product": {
        "emphasis": "Problem framing (GDP cost, annual deaths), solution narrative (targeted interventions at blackspots), limitations transparency, upgrade path documented",
        "link": "https://soochak.onrender.com",
        "line": "Built a tool that translates ML predictions into infrastructure investment priorities — explains which road features drive fatality risk and simulates the impact of specific interventions.",
        "q": "What's the business case? — India spends ₹5.96L crore/year on accident costs. Targeting top 5% high-risk blackspots with dividers/signals has highest ROI; SOOCHAK quantifies which features matter most."
    },
}

selected_role = st.selectbox("Select your role:", list(roles.keys()))

if selected_role:
    role = roles[selected_role]
    st.subheader(f"🎯 {selected_role} Angle")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**What to emphasize:** {role['emphasis']}")
        st.markdown(f"**Opening line:** *{role['line']}*")
        st.markdown(f"**Likely interview Q:** {role['q']}")
    with col2:
        st.markdown(f"[📂 View relevant code →]({role['link']})")
        st.markdown("[🔴 Live demo →](https://soochak.onrender.com)")
        st.markdown("[📖 GitHub →](https://github.com/GokavalasaHemanthNaidu/Soochak)")

st.divider()
st.caption("SOOCHAK v1.0.0 — F1: 0.47 | AUC: 0.72 | Threshold: 0.18 | Dataset: India FIR | Deployment: Render Free Tier")
