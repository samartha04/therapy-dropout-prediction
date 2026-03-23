import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from src.evaluate import compute_risk_score
import uvicorn


load_dotenv()


class PatientSession(BaseModel):
    """
    Session-level clinical and engagement information for a psychotherapy patient.

    These features capture current symptom severity (PHQ-9), treatment dose
    (session number and frequency), attendance patterns, gaps between sessions,
    mood, age, and the rate of change in depressive symptoms. Together, they
    provide the model with a snapshot of dropout risk at this point in care.
    """

    phq9_score: int = Field(..., ge=0, le=27)
    session_number: int = Field(..., ge=1, le=20)
    session_frequency_per_month: float = Field(..., ge=1.0, le=8.0)
    attendance_consistency: float = Field(..., ge=0.0, le=1.0)
    gap_between_sessions_days: int = Field(..., ge=1, le=90)
    mood_rating: int = Field(..., ge=1, le=10)
    age: int = Field(..., ge=18, le=70)
    phq9_change_rate: float = Field(..., ge=-5.0, le=5.0)


class PredictionResponse(BaseModel):
    """
    Model-derived dropout risk estimate and human-readable explanation.

    Risk is expressed both as a 0–100 probability-style score and as a
    categorical tier (Low, Moderate, High) that clinicians can interpret
    quickly. The message field provides a brief plain-language summary that
    contextualizes the numeric risk for shared decision making.
    """

    risk_score: float = Field(..., ge=0.0, le=100.0)
    risk_tier: str
    message: str


def _build_plain_english_message(risk_score: float, risk_tier: str) -> str:
    """
    Convert numeric dropout risk into a concise clinical explanation.

    Parameters
    ----------
    risk_score : float
        Dropout risk on a 0–100 scale.
    risk_tier : str
        Tier label ('Low', 'Moderate', or 'High').

    Returns
    -------
    str
        Human-readable summary of the model's estimate.
    """
    if risk_tier == "Low":
        return (
            f"The model estimates a low risk of early dropout ({risk_score:.1f}%). "
            "Engagement appears generally stable, but ongoing monitoring is still recommended."
        )
    if risk_tier == "Moderate":
        return (
            f"The model estimates a moderate risk of early dropout ({risk_score:.1f}%). "
            "Consider exploring barriers to attendance and reinforcing engagement."
        )
    if risk_tier == "High":
        return (
            f"The model estimates a high risk of early dropout ({risk_score:.1f}%). "
            "Proactive outreach, safety planning, or care coordination may be warranted."
        )
    return (
        f"The model estimates a dropout risk of approximately {risk_score:.1f}%. "
        "Please interpret this in the context of the full clinical picture."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context for loading and managing the dropout model.

    The model is loaded once at startup from a path specified in environment
    variables, and stored on the application state. This avoids repeated disk
    access and ensures that the API can quickly serve risk predictions for
    psychotherapy sessions.
    """
    model_path = os.getenv("DROPOUT_MODEL_PATH")
    app.state.model = None
    app.state.model_loaded = False

    if model_path:
        try:
            with open(model_path, "rb") as f:
                app.state.model = pickle.load(f)
            app.state.model_loaded = True
        except FileNotFoundError:
            app.state.model = None
            app.state.model_loaded = False
        except Exception:
            app.state.model = None
            app.state.model_loaded = False

    yield

    app.state.model = None
    app.state.model_loaded = False


app = FastAPI(
    title="Psychotherapy Dropout Risk API",
    version="1.0.0",
    description=(
        "API for predicting early psychotherapy dropout risk based on session-level "
        "clinical and engagement features. Intended for research and decision support only."
    ),
    lifespan=lifespan,
)


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic API metadata and model status.

    Returns the API name, version, and a simple status flag, along with
    whether the dropout prediction model has been successfully loaded into
    memory. This is useful for quick connectivity checks.
    """
    return {
        "name": app.title,
        "version": app.version,
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring API and model readiness.

    Returns a simple health status string and an explicit boolean indicating
    whether the dropout prediction model is currently available. This can be
    integrated into monitoring or orchestration tools.
    """
    model_loaded = bool(getattr(app.state, "model_loaded", False))
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(session: PatientSession) -> PredictionResponse:
    """
    Predict early dropout risk for a single psychotherapy session.

    This endpoint accepts session-level clinical features for a patient,
    runs the dropout prediction model, and returns a probabilistic risk
    score, a risk tier, and a brief plain-language explanation to support
    collaborative decision making between clinicians and patients.
    """
    model = getattr(app.state, "model", None)
    model_loaded = bool(getattr(app.state, "model_loaded", False))

    if not model_loaded or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is not loaded. Please try again later.",
        )

    features = {
        "phq9_score": session.phq9_score,
        "session_number": session.session_number,
        "session_frequency_per_month": session.session_frequency_per_month,
        "attendance_consistency": session.attendance_consistency,
        "gap_between_sessions_days": session.gap_between_sessions_days,
        "mood_rating": session.mood_rating,
        "age": session.age,
        "phq9_change_rate": session.phq9_change_rate,
    }
    patient_df = __import__("pandas").DataFrame([features])

    risk_result = compute_risk_score(model, patient_df)
    risk_score = float(risk_result["risk_score"])
    risk_tier = str(risk_result["risk_tier"])
    message = _build_plain_english_message(risk_score, risk_tier)

    return PredictionResponse(
        risk_score=risk_score,
        risk_tier=risk_tier,
        message=message,
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
    )

