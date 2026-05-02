import os
from typing import List, Optional

import numpy as np
import pandas as pd


def generate_synthetic_data(n_patients: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate clinically informed synthetic psychotherapy data.
    Dropout is correlated with features based on published literature:
    - Higher PHQ-9 score increases dropout risk (Swift & Greenberg 2012)
    - Lower attendance increases dropout risk (McGovern 2024)
    - Longer session gaps increase dropout risk (Lutz et al 2022)
    - Lower mood increases dropout risk
    - Earlier sessions increase dropout risk
    """
    np.random.seed(seed)
    n = n_patients

    phq9_score = np.random.randint(0, 27, n)
    session_number = np.random.randint(1, 20, n)
    session_frequency = np.random.uniform(1, 8, n)
    attendance = np.random.uniform(0, 1, n)
    gap_days = np.random.randint(3, 60, n)
    mood = np.random.randint(1, 10, n)
    age = np.random.randint(18, 70, n)
    phq9_change = np.random.uniform(-2, 2, n)

    log_odds = (
          1.0
        + 0.20 * phq9_score
        - 8.0 * attendance
        + 0.10 * gap_days
        - 0.40 * mood
        - 0.15 * session_number
        + 1.20 * phq9_change
        - 0.30 * session_frequency
    )

    prob = 1 / (1 + np.exp(-log_odds))
    dropout = np.random.binomial(1, prob)

    data = pd.DataFrame({
        'patient_id': range(1, n + 1),
        'phq9_score': phq9_score,
        'session_number': session_number,
        'session_frequency_per_month': session_frequency,
        'attendance_consistency': attendance,
        'gap_between_sessions_days': gap_days,
        'mood_rating': mood,
        'age': age,
        'phq9_change_rate': phq9_change,
        'dropout': dropout
    })

    print(f"Shape: {data.shape}")
    print(f"Dropout rate: {data['dropout'].mean():.2%}")
    return data

def get_synthetic_feature_columns() -> List[str]:
    """
    Return feature column names for the synthetic psychotherapy dropout dataset.

    These variables are commonly cited in the psychotherapy dropout literature as
    predictors or correlates of premature termination (e.g., symptom severity,
    treatment dose, attendance patterns, and short-term symptom trajectory).

    Returns
    -------
    list of str
        Feature columns to use for modeling (excludes identifiers and the target).
    """
    return [
        "phq9_score",
        "session_number",
        "session_frequency_per_month",
        "attendance_consistency",
        "gap_between_sessions_days",
        "mood_rating",
        "age",
        "phq9_change_rate",
    ]


def load_daic_woz(raw_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load DAIC-WOZ (Distress Analysis Interview Corpus) depression splits for reference.

    IMPORTANT:
    DAIC-WOZ is a depression detection benchmark, not a psychotherapy dropout dataset.
    This loader is kept in the codebase to support *feature selection validation*
    and documentation, not for training the psychotherapy dropout model.

    Parameters
    ----------
    raw_dir : str, optional
        Directory containing ``train_split_Depression_AVEC2017.csv``,
        ``dev_split_Depression_AVEC2017.csv``, and ``test_split_Depression_AVEC2017.csv``.
        Defaults to ``data/raw`` relative to this package (project root layout).

    Returns
    -------
    pd.DataFrame
        Stacked train/dev/test rows with a ``split`` column and renamed targets /
        demographics; original PHQ item columns are retained.
    """
    if raw_dir is None:
        raw_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        )

    train_path = os.path.join(raw_dir, "train_split_Depression_AVEC2017.csv")
    dev_path = os.path.join(raw_dir, "dev_split_Depression_AVEC2017.csv")
    test_path = os.path.join(raw_dir, "test_split_Depression_AVEC2017.csv")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.assign(split="train")
    dev_df = dev_df.assign(split="dev")
    test_df = test_df.assign(split="test")

    combined = pd.concat([train_df, dev_df, test_df], axis=0, ignore_index=True)

    combined = combined.rename(
        columns={
            "PHQ8_Score": "phq9_score",
            "PHQ8_Binary": "dropout",
            "Gender": "gender",
        }
    )

    print(f"DAIC-WOZ combined shape: {combined.shape}")
    check_class_balance(combined, target_col="dropout")

    return combined


def get_feature_columns() -> List[str]:
    """
    Return model feature names for DAIC-WOZ depression / dropout prediction.

    Excludes identifiers (``Participant_ID``), the binary outcome (``dropout``),
    and the data split indicator (``split``). Features are the total symptom score,
    gender, and each PHQ-8 item score available in the AVEC 2017 depression splits.

    Returns
    -------
    list of str
        Fixed column order for training and evaluation pipelines.
    """
    return [
        "phq9_score",
        "gender",
        "PHQ8_NoInterest",
        "PHQ8_Depressed",
        "PHQ8_Sleep",
        "PHQ8_Tired",
        "PHQ8_Appetite",
        "PHQ8_Failure",
        "PHQ8_Concentrating",
        "PHQ8_Moving",
    ]


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
