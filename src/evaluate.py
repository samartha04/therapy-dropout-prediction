import os
import pickle
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import shap


def compute_shap_values(
    model: Any, X_test: pd.DataFrame
) -> Tuple[shap.Explainer, shap.Explanation]:
    """
    Compute SHAP values for dropout risk predictions on the test set.

    Clinically, SHAP values decompose the model's dropout probability into
    feature-level contributions for each patient. This helps explain how
    factors such as symptom severity, engagement patterns, and demographics
    push a given patient's risk of premature termination up or down.

    Parameters
    ----------
    model : Any
        Trained dropout prediction model (e.g., XGBoost classifier).
    X_test : pd.DataFrame
        Test feature matrix containing the clinical and behavioral predictors.

    Returns
    -------
    explainer : shap.Explainer
        SHAP explainer object tied to the trained model.
    shap_values : shap.Explanation
        SHAP values for all patients in the test set.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values


def plot_global_importance(
    shap_values: shap.Explanation, X_test: pd.DataFrame
) -> None:
    """
    Plot global feature importance using a SHAP summary plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values for the test set.
    X_test : pd.DataFrame
        Test feature matrix corresponding to the SHAP values.
    """
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Global SHAP summary — impact on dropout risk")
    plt.xlabel("SHAP value (impact on model output)")
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "global_importance.png"), dpi=300)
    plt.close()


def plot_individual_explanation(
    explainer: shap.Explainer,
    shap_values: shap.Explanation,
    X_test: pd.DataFrame,
    patient_index: int,
) -> None:
    """
    Plot an individual SHAP waterfall explanation for a single patient.

    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer object associated with the trained model.
    shap_values : shap.Explanation
        SHAP values for the full test set.
    X_test : pd.DataFrame
        Test feature matrix.
    patient_index : int
        Index of the patient in X_test for whom to generate the explanation.
    """
    os.makedirs("reports", exist_ok=True)

    if patient_index < 0 or patient_index >= len(X_test):
        raise IndexError("patient_index is out of bounds for X_test.")

    readable_names: Dict[str, str] = {
        "phq9_score": "PHQ-9 Depression Score",
        "session_number": "Session Number",
        "session_frequency_per_month": "Sessions Per Month",
        "attendance_consistency": "Attendance Consistency",
        "gap_between_sessions_days": "Days Between Sessions",
        "mood_rating": "Mood Rating",
        "age": "Patient Age",
        "phq9_change_rate": "PHQ-9 Change Rate",
        "gap_increasing": "Increasing Session Gaps",
        "max_attendance_streak": "Longest Attendance Streak",
        "phq9_change_rate_abs": "PHQ-9 Change Magnitude",
    }

    patient_row = X_test.iloc[patient_index]
    renamed_index = [readable_names.get(col, col) for col in patient_row.index]
    patient_row_readable = pd.Series(patient_row.values, index=renamed_index)

    base_value = shap_values.base_values[patient_index]
    patient_shap_values = shap_values.values[patient_index]

    patient_explanation = shap.Explanation(
        values=patient_shap_values,
        base_values=base_value,
        data=patient_row_readable.values,
        feature_names=list(patient_row_readable.index),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(patient_explanation, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join("reports", f"patient_{patient_index}_explanation.png"),
        dpi=300,
    )
    plt.close()


def compute_risk_score(model: Any, patient_features: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a clinical dropout risk score and tier for a single patient.

    Loads the saved StandardScaler to match training preprocessing before
    passing features to the model. This ensures predictions are on the
    same scale as training data.

    Parameters
    ----------
    model : Any
        Trained dropout prediction model supporting predict_proba.
    patient_features : pd.DataFrame
        Single-row DataFrame containing the patient's raw features.

    Returns
    -------
    dict
        Dictionary with risk_score (0-100) and risk_tier (Low/Moderate/High).
    """
    if patient_features.shape[0] != 1:
        raise ValueError("patient_features must contain exactly one patient.")

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must implement predict_proba.")

    scaler_path = os.path.join("models", "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        features_scaled = scaler.transform(patient_features)
    else:
        features_scaled = patient_features.values

    proba = float(model.predict_proba(features_scaled)[0][1])
    risk_score = round(proba * 100, 2)

    if risk_score <= 33:
        risk_tier = "Low"
    elif risk_score <= 66:
        risk_tier = "Moderate"
    else:
        risk_tier = "High"

    return {"risk_score": risk_score, "risk_tier": risk_tier}
    