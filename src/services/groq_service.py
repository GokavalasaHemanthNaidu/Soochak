"""Async Groq API client for LLM explanations with rule-based fallback."""

import os
import json
import httpx
from typing import Optional

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3-70b-8192"
TIMEOUT = 10.0


async def explain_prediction(
    top_features: list,
    probability: float,
    predicted_class: int
) -> Optional[str]:
    """Generate plain-English explanation using Groq LLM with fallback."""
    if not GROQ_API_KEY:
        return _generate_fallback(top_features, probability, predicted_class)

    severity_text = "FATAL" if predicted_class == 1 else "NON-FATAL"

    prompt = f"""You are a road safety expert analyzing an accident prediction model.

Prediction: {severity_text} accident with {probability:.1%} probability.

Top risk factors (SHAP values):
{json.dumps(top_features, indent=2)}

Write a 2-3 sentence plain-English explanation of why this accident is predicted as {severity_text}.
Include a specific policy recommendation for reducing risk.
Be concise and actionable."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(2):
            try:
                response = await client.post(GROQ_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                if attempt == 1:
                    break
                continue

    return _generate_fallback(top_features, probability, predicted_class)


def _generate_fallback(top_features, probability, predicted_class) -> str:
    """Generate rule-based explanation when Groq is unavailable."""
    severity = "fatal" if predicted_class == 1 else "non-fatal"
    top = top_features[0]["feature"].lower() if top_features else "unknown"

    explanations = {
        "road_type": f"This accident is predicted as {severity} because of the road type. National highways without dividers have significantly higher fatality rates.",
        "vehicle_type": f"The vehicle type is the strongest risk factor. Two-wheelers lack structural protection, making this accident likely {severity}.",
        "speed_limit": f"High speed limits reduce reaction time. This contributes to the {severity} prediction.",
        "lighting": f"Poor lighting conditions increase accident severity. This is a key factor in the {severity} prediction.",
        "weather": f"Adverse weather reduces visibility and road grip, contributing to the {severity} prediction.",
        "junction": f"Junction geometry affects collision angles. This is a factor in the {severity} prediction.",
        "junction_ctrl": f"Lack of traffic control at junctions increases conflict points, contributing to the {severity} prediction.",
        "driver_age": f"Driver age affects risk profile. This is a factor in the {severity} prediction.",
        "urban_rural": f"Rural roads have higher speeds and fewer emergency services, contributing to the {severity} prediction.",
    }

    base = explanations.get(top, f"The model predicts this accident as {severity} with {probability:.1%} confidence based on multiple risk factors.")
    return base + " Recommendation: Improve infrastructure and enforce speed limits in this segment."
