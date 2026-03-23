import numpy as np
import pandas as pd


def generate_synthetic_data(n_patients: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic psychotherapy outcome data for a cohort of patients.

    This function simulates clinically plausible data for patients in psychotherapy,
    including depression severity (PHQ-9), treatment engagement (session number,
    frequency, attendance consistency, gaps between sessions), subjective mood ratings,
    age, and rate of change in PHQ-9 scores over time. It also generates a binary
    dropout indicator, where 0 represents patients who remain in treatment and 1
    represents patients who discontinue prematurely (roughly 35% dropout rate).

    Parameters
    ----------
    n_patients : int, optional
        Number of unique patients to simulate, by default 500.
    seed : int, optional
        Random seed for reproducibility of the synthetic cohort, by default 42.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per patient and the following columns:
        - patient_id
        - phq9_score (0–27, higher = more severe depression)
        - session_number (1–20, later sessions indicate greater treatment exposure)
        - session_frequency_per_month (1–8, approximate sessions per month)
        - attendance_consistency (0–1, fraction of scheduled sessions attended)
        - gap_between_sessions_days (3–60, days between most recent sessions)
        - mood_rating (1–10, higher = better mood)
        - age (18–70, patient age in years)
        - phq9_change_rate (-2 to 2, change per unit time, negative = improvement)
        - dropout (0 = completed/ongoing, 1 = prematurely stopped therapy)
    """
    rng = np.random.default_rng(seed)

    patient_id = np.arange(1, n_patients + 1)

    # Baseline PHQ-9: skewed toward mild–moderate depression but spanning full range
    phq9_score = rng.integers(low=0, high=28, size=n_patients)

    # Session exposure: patients at various stages of treatment
    session_number = rng.integers(low=1, high=21, size=n_patients)

    # Frequency of sessions: typical outpatient frequencies between monthly and twice weekly
    session_frequency_per_month = rng.uniform(low=1.0, high=8.0, size=n_patients)

    # Attendance consistency: proportion of attended sessions
    attendance_consistency = rng.uniform(low=0.0, high=1.0, size=n_patients)

    # Gaps between sessions: larger gaps may reflect inconsistent engagement
    gap_between_sessions_days = rng.integers(low=3, high=61, size=n_patients)

    # Subjective mood ratings: simple ordinal scale where higher is better
    mood_rating = rng.integers(low=1, high=11, size=n_patients)

    # Age distribution: adult outpatient range
    age = rng.integers(low=18, high=71, size=n_patients)

    # Rate of change in PHQ-9: negative implies clinical improvement over time
    phq9_change_rate = rng.uniform(low=-2.0, high=2.0, size=n_patients)

    # Dropout indicator: imbalanced with ~65% non-dropout (0) and ~35% dropout (1)
    dropout = rng.choice([0, 1], size=n_patients, p=[0.65, 0.35])

    data = pd.DataFrame(
        {
            "patient_id": patient_id,
            "phq9_score": phq9_score,
            "session_number": session_number,
            "session_frequency_per_month": session_frequency_per_month,
            "attendance_consistency": attendance_consistency,
            "gap_between_sessions_days": gap_between_sessions_days,
            "mood_rating": mood_rating,
            "age": age,
            "phq9_change_rate": phq9_change_rate,
            "dropout": dropout,
        }
    )

    return data


def load_real_data(filepath: str) -> pd.DataFrame:
    """
    Load real psychotherapy dataset from a CSV file.

    This function is intended for clinical or research datasets containing
    psychotherapy engagement and outcome variables (e.g., PHQ-9 scores,
    session information, and dropout labels) stored in a CSV file. It does
    not assume any particular schema beyond being a valid CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing real-world psychotherapy data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded clinical data for further analysis
        and modeling (e.g., dropout prediction or outcome evaluation).
    """
    df = pd.read_csv(filepath)
    return df


def check_class_balance(df: pd.DataFrame, target_col: str) -> None:
    """
    Inspect and report class balance for a target clinical outcome.

    This function prints both the raw counts and percentages of each class
    in a specified target column, which is typically a clinically meaningful
    outcome such as treatment dropout (0 vs. 1) or response/remission status.
    Understanding class balance is important for modeling dropout risk and
    making sure that classifiers are not biased toward the majority group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing psychotherapy data, including the target column.
    target_col : str
        Name of the target column whose class distribution should be examined.

    Returns
    -------
    None
        The function prints summary statistics and does not return a value.
    """
    counts = df[target_col].value_counts(dropna=False)
    percentages = df[target_col].value_counts(normalize=True, dropna=False) * 100

    print("Class balance for target column:", target_col)
    for cls in counts.index:
        count = counts[cls]
        pct = percentages[cls]
        print(f"  Class {cls}: {count} samples ({pct:.2f}%)")

