import numpy as np
import pandas as pd


def compute_phq9_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the per-session rate of change in PHQ-9 scores for each patient.

    Clinically, PHQ-9 measures depression severity, and the rate at which this score
    changes across sessions captures how quickly a patient is improving or worsening.
    A negative change rate indicates that PHQ-9 scores are decreasing over sessions,
    which corresponds to clinical improvement. A positive rate suggests deterioration
    or insufficient response to psychotherapy.

    This function expects a longitudinal DataFrame with at least the columns:
    - 'patient_id': unique identifier for each patient
    - 'session_number': ordinal index of the therapy session (increasing over time)
    - 'phq9_score': PHQ-9 score recorded at each session

    Parameters
    ----------
    df : pd.DataFrame
        Longitudinal psychotherapy data with per-session PHQ-9 scores.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with an additional column:
        - 'phq9_change_rate': per-session change in PHQ-9 score for each patient,
          broadcast to all that patient's rows (negative = improvement).
    """
    df_copy = df.copy(deep=True)

    if not {"patient_id", "session_number", "phq9_score"}.issubset(df_copy.columns):
        raise ValueError(
            "DataFrame must contain 'patient_id', 'session_number', and 'phq9_score' columns."
        )

    # Ensure sessions are ordered within each patient
    df_copy = df_copy.sort_values(["patient_id", "session_number"])

    def _per_patient_change_rate(group: pd.DataFrame) -> float:
        first_score = group["phq9_score"].iloc[0]
        last_score = group["phq9_score"].iloc[-1]
        first_session = group["session_number"].iloc[0]
        last_session = group["session_number"].iloc[-1]

        # Number of sessions spanned; at least 1 to avoid division by zero
        session_span = max(1, last_session - first_session)
        return (last_score - first_score) / session_span

    rates = (
        df_copy.groupby("patient_id", group_keys=False)
        .apply(_per_patient_change_rate)
        .rename("phq9_change_rate")
    )

    # Map per-patient rate back to each row
    df_copy["phq9_change_rate"] = df_copy["patient_id"].map(rates)

    return df_copy


def compute_session_gap_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag patients whose gaps between sessions show an increasing pattern over time.

    In psychotherapy, progressively longer gaps between sessions often signal
    disengagement and a higher risk of dropout. This function examines the
    trajectory of 'gap_between_sessions_days' within each patient and identifies
    those whose gaps tend to increase over time, marking them as higher-risk
    based on their attendance pattern.

    This function expects a longitudinal DataFrame with at least the columns:
    - 'patient_id': unique identifier for each patient
    - 'session_number': ordinal index of the therapy session (increasing over time)
    - 'gap_between_sessions_days': days between consecutive sessions (per row)

    Parameters
    ----------
    df : pd.DataFrame
        Longitudinal psychotherapy data with per-session gaps between appointments.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with an additional column:
        - 'gap_increasing': 1 if the overall pattern of session gaps for the patient
          is increasing (early warning for dropout), otherwise 0.
    """
    df_copy = df.copy(deep=True)

    if not {"patient_id", "session_number", "gap_between_sessions_days"}.issubset(
        df_copy.columns
    ):
        raise ValueError(
            "DataFrame must contain 'patient_id', 'session_number', and "
            "'gap_between_sessions_days' columns."
        )

    df_copy = df_copy.sort_values(["patient_id", "session_number"])

    def _flag_increasing_gaps(group: pd.DataFrame) -> int:
        gaps = group["gap_between_sessions_days"].values

        # If there are fewer than 3 sessions, treat as not enough evidence of a pattern
        if gaps.size < 3:
            return 0

        # Use sign of the mean first difference as a simple increasing/decreasing indicator
        diffs = np.diff(gaps.astype(float))
        mean_diff = np.mean(diffs)

        # Flag as increasing if average change is meaningfully positive
        return int(mean_diff > 0)

    flags = (
        df_copy.groupby("patient_id", group_keys=False)
        .apply(_flag_increasing_gaps)
        .rename("gap_increasing")
    )

    df_copy["gap_increasing"] = df_copy["patient_id"].map(flags).astype(int)

    return df_copy


def compute_attendance_streak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the longest consecutive attendance streak for each patient.

    Clinically, consistent attendance at psychotherapy sessions is strongly
    associated with better outcomes and lower dropout risk. This function
    summarizes each patient's engagement pattern by calculating their longest
    run of consecutively attended sessions, which can serve as a protective
    factor feature in dropout prediction models.

    This function expects a longitudinal DataFrame with at least the columns:
    - 'patient_id': unique identifier for each patient
    - 'session_number': ordinal index of the therapy session (increasing over time)
    - 'attended': binary indicator (1 = attended, 0 = missed) for each scheduled session

    Parameters
    ----------
    df : pd.DataFrame
        Longitudinal psychotherapy data with a per-session attendance indicator.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with an additional column:
        - 'max_attendance_streak': longest run of consecutively attended sessions
          (value repeated for all rows of a given patient).
    """
    df_copy = df.copy(deep=True)

    if not {"patient_id", "session_number", "attended"}.issubset(df_copy.columns):
        raise ValueError(
            "DataFrame must contain 'patient_id', 'session_number', and 'attended' columns."
        )

    df_copy = df_copy.sort_values(["patient_id", "session_number"])

    def _longest_streak(group: pd.DataFrame) -> int:
        attended_values = group["attended"].astype(int).values

        max_streak = 0
        current_streak = 0

        for val in attended_values:
            if val == 1:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0

        return max_streak

    streaks = (
        df_copy.groupby("patient_id", group_keys=False)
        .apply(_longest_streak)
        .rename("max_attendance_streak")
    )

    df_copy["max_attendance_streak"] = df_copy["patient_id"].map(streaks).astype(int)

    return df_copy


def run_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full clinical feature engineering pipeline for dropout prediction.

    This function sequentially applies several clinically motivated feature
    transformations to psychotherapy session data:
    1. PHQ-9 change rate: how quickly depression severity improves or worsens.
    2. Session gap pattern: whether the time between sessions is increasing,
       which may signal early disengagement.
    3. Attendance streak: the longest stretch of consistent attendance, capturing
       sustained engagement in treatment.

    These features are designed to help machine learning models capture
    clinically meaningful risk and resilience factors related to psychotherapy
    dropout.

    Parameters
    ----------
    df : pd.DataFrame
        Longitudinal psychotherapy dataset containing at least:
        - 'patient_id', 'session_number', 'phq9_score',
          'gap_between_sessions_days', and 'attended'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with all engineered features added:
        - 'phq9_change_rate'
        - 'gap_increasing'
        - 'max_attendance_streak'
    """
    df_features = compute_phq9_change_rate(df)
    df_features = compute_session_gap_pattern(df_features)
    df_features = compute_attendance_streak(df_features)

    return df_features

