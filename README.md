# SOOCHAK

## Overview
SOOCHAK is an Explainable Road Accident Severity Predictor + Intervention Simulator. It predicts fatal road accidents using real FIR data, explains the predictions using SHAP and Groq, and simulates what-if scenarios (e.g., adding a divider). Deployed live on Render.

## Tech Stack
- **Frontend**: HTML/JS with Jinja2 Templates (served via FastAPI)
- **Backend**: Python 3.11+, FastAPI, Uvicorn, aiosqlite
- **Machine Learning**: XGBoost, ONNX Runtime, SHAP, Optuna, Scikit-learn
- **Database**: SQLite (Async)
- **LLM / Explanations**: Groq API

## How to Run
1. Create and activate the virtual environment: 
   `python -m venv venv`
   `.\venv\Scripts\activate`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Set up the database:
   `mkdir -p data`
4. Run the application:
   `python -m src.api.main`
   Visit `http://localhost:8000` for the dashboard.

## Project Context for AI Agents
**Agents:** Start by reading `TODO.md` to understand the current state, and `AGENT_RULES.md` to understand the coding standards.
