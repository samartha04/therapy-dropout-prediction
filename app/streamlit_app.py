import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from typing import Any, Dict

import matplotlib.pyplot as plt  # noqa: F401
import pandas as pd
import streamlit as st

from src.evaluate import (
    compute_risk_score,
    compute_shap_values,
    plot_individual_explanation,
)

MODEL_PATH = "models/xgboost_model.pkl"


def load_model() -> Any:
    """
    Load the pretrained psychotherapy dropout prediction model.

    The model encapsulates patterns learned from historical psychotherapy data,
    such as how symptom severity and engagement behaviors relate to early
    termination. Loading it once at startup avoids repeated disk access and
    keeps the app responsive for clinicians exploring risk scenarios.

    Returns
    -------
    Any
        The loaded model object.

    Raises
    ------
    FileNotFoundError
        If the expected model file is not found on disk, indicating that the
        training pipeline has not been run or the artifact path is incorrect.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please train and save the model before using the app."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_resource(show_spinner=False)
def get_model() -> Any:
    """
    Cached accessor for the pretrained dropout prediction model.

    This function ensures the model is loaded only once per Streamlit session,
    so repeated predictions for different what-if scenarios remain fast.

    Returns
    -------
    Any
        The cached model instance.
    """
    return load_model()


def build_patient_features(
    phq9_score: int,
    session_number: int,
    sessions_per_month: float,
    attendance_consistency: float,
    days_since_last_session: int,
    mood_rating: int,
    patient_age: int,
    phq9_change_rate: float,
) -> pd.DataFrame:
    """
    Construct a single-patient feature row from UI inputs including
    engineered features to match the training pipeline.

    Parameters
    ----------
    phq9_score : int
        Current PHQ-9 depression severity score (0-27).
    session_number : int
        Current session index in the treatment (1-20).
    sessions_per_month : float
        Typical number of sessions per month (frequency of care).
    attendance_consistency : float
        Proportion of scheduled sessions attended (0-1).
    days_since_last_session : int
        Days since the most recent therapy session.
    mood_rating : int
        Subjective mood rating (1-10), higher is better.
    patient_age : int
        Patient age in years.
    phq9_change_rate : float
        Rate of change in PHQ-9 per session (negative = improving).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame formatted to match the model's expected features.
    """
    features: Dict[str, Any] = {
        "phq9_score": phq9_score,
        "session_number": session_number,
        "session_frequency_per_month": sessions_per_month,
        "attendance_consistency": attendance_consistency,
        "gap_between_sessions_days": days_since_last_session,
        "mood_rating": mood_rating,
        "age": patient_age,
        "phq9_change_rate": phq9_change_rate,
        # Engineered features — must match training pipeline
        "gap_increasing": 1 if days_since_last_session > 14 else 0,
        "max_attendance_streak": int(attendance_consistency * session_number),
        "phq9_change_rate_abs": abs(phq9_change_rate),
    }
    return pd.DataFrame([features])


def display_risk_result(risk_score: float, risk_tier: str) -> None:
    """
    Display dropout risk score, tier, and visual gauge in the main panel.

    Parameters
    ----------
    risk_score : float
        Dropout risk score between 0 and 100.
    risk_tier : str
        Risk category: 'Low', 'Moderate', or 'High'.

    Returns
    -------
    None
    """
    st.subheader("Predicted Dropout Risk")

    st.markdown(
        f"<h1 style='text-align: center; font-size: 3rem;'>{risk_score:.1f}%</h1>",
        unsafe_allow_html=True,
    )

    color_map = {
        "Low": "#22c55e",
        "Moderate": "#eab308",
        "High": "#ef4444",
    }
    badge_color = color_map.get(risk_tier, "#6b7280")

    st.markdown(
        f"""
        <p style="
            text-align: center;
            margin-top: 0.5rem;
        ">
            <span style="
                background-color: {badge_color};
                color: white;
                padding: 0.4rem 1.2rem;
                border-radius: 999px;
                font-weight: 600;
            ">
                {risk_tier} Risk
            </span>
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Risk Gauge")
    st.progress(min(max(risk_score / 100.0, 0.0), 1.0))


def show_shap_explanation(model: Any, patient_features: pd.DataFrame) -> None:
    """
    Generate and display a SHAP waterfall explanation for the current patient.

    Parameters
    ----------
    model : Any
        Trained dropout prediction model.
    patient_features : pd.DataFrame
        Single-row DataFrame representing the patient's features.

    Returns
    -------
    None
    """
    explainer, shap_values = compute_shap_values(model, patient_features)
    plot_individual_explanation(
        explainer=explainer,
        shap_values=shap_values,
        X_test=patient_features,
        patient_index=0,
    )

    shap_path = os.path.join("reports", "patient_0_explanation.png")
    if os.path.exists(shap_path):
        st.subheader("Why this risk score?")
        st.image(
            shap_path,
            caption="SHAP Waterfall Explanation (feature contributions to dropout risk)",
            use_container_width=True,
        )
    else:
        st.warning(
            "Could not locate the SHAP explanation image. "
            "Please check that the reports directory is writable."
        )


def main() -> None:
    """
    Streamlit app for early psychotherapy dropout risk prediction.
    """
    st.set_page_config(
        page_title="Psychotherapy Dropout Risk",
        layout="wide",
        page_icon="🧠",
    )

    st.title("Early Dropout Risk Prediction in Psychotherapy")

    with st.sidebar:
        st.header("Patient Inputs")

        phq9_score = st.slider("PHQ-9 Score", min_value=0, max_value=27, value=12)
        session_number = st.slider("Session Number", min_value=1, max_value=20, value=3)
        sessions_per_month = st.slider(
            "Sessions Per Month",
            min_value=1.0,
            max_value=8.0,
            value=4.0,
            step=0.5,
        )
        attendance_consistency = st.slider(
            "Attendance Consistency",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
        )
        days_since_last_session = st.slider(
            "Days Since Last Session",
            min_value=1,
            max_value=90,
            value=14,
        )
        mood_rating = st.slider("Mood Rating", min_value=1, max_value=10, value=6)
        patient_age = st.slider("Patient Age", min_value=18, max_value=70, value=30)
        phq9_change_rate = st.slider(
            "PHQ-9 Change Rate",
            min_value=-5.0,
            max_value=5.0,
            value=-0.5,
            step=0.1,
        )

        predict_button = st.button("Predict Risk", type="primary")

    if predict_button:
        try:
            model = get_model()
        except FileNotFoundError:
            st.error(
                "Model file not found. Please run the training pipeline and save "
                f"an XGBoost model to '{MODEL_PATH}' before using this app."
            )
            st.stop()

        patient_features = build_patient_features(
            phq9_score=phq9_score,
            session_number=session_number,
            sessions_per_month=sessions_per_month,
            attendance_consistency=attendance_consistency,
            days_since_last_session=days_since_last_session,
            mood_rating=mood_rating,
            patient_age=patient_age,
            phq9_change_rate=phq9_change_rate,
        )

        with st.spinner("Computing risk score and explanation..."):
            risk_result = compute_risk_score(model, patient_features)
            risk_score = risk_result["risk_score"]
            risk_tier = risk_result["risk_tier"]

            display_risk_result(risk_score, risk_tier)
            show_shap_explanation(model, patient_features)

    st.markdown("---")
    st.markdown(
        """
        **Ethical Disclaimer**

        This is a research prototype. Not intended for clinical use.  
        All predictions are decision support only.
        """
    )


if __name__ == "__main__":
    main()