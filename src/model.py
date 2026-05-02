from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_loader import get_feature_columns


def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Resolve feature columns for synthetic psychotherapy dropout data.
    Excludes target, identifier and split columns.
    """
    exclude = {'dropout', 'patient_id', 'split'}
    return [c for c in df.columns if c not in exclude]


def _should_use_official_split(df: pd.DataFrame) -> bool:
    """Use AVEC train+dev vs test when a split column from load_daic_woz() is present."""
    if "split" not in df.columns:
        return False
    s = df["split"].astype(str)
    return bool(s.eq("test").any() and (s.eq("train") | s.eq("dev")).any())


def prepare_data(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    use_official_split: Optional[bool] = None,
) -> Tuple[Any, Any, Any, Any, StandardScaler]:
    """
    Prepare psychotherapy dropout dataset for model training and evaluation.

    This function takes the full dataset, separates features from the clinical
    target label (dropout), splits the data into training and test sets, and
    applies standardization to the feature space. Standardizing features helps
    many models (especially linear ones) interpret clinical variables like
    symptom scores and engagement metrics on a comparable scale.

    For DAIC-WOZ data loaded with ``load_daic_woz()``, feature columns default to
    ``get_feature_columns()`` (PHQ total, gender, PHQ-8 items) and, when a
    ``split`` column is present, train+dev rows are used for fitting the scaler
    and training while the held-out ``test`` split is used for evaluation—matching
    the official AVEC partition. Other datasets keep the prior behavior: every
    column except ``dropout``, with a random 80/20 stratified split.

    The target column is assumed to be:
    - 'dropout': 0 for patients who complete or continue treatment,
                 1 for patients who discontinue therapy prematurely.

    Parameters
    ----------
    df : pd.DataFrame
        Full psychotherapy dataset including the 'dropout' target column and
        any engineered clinical features.
    feature_columns : sequence of str, optional
        Explicit feature names. If omitted, resolved via ``get_feature_columns()``
        when those columns exist in ``df``; otherwise all columns except ``dropout``.
    use_official_split : bool, optional
        If True, use ``split`` in ``{'train','dev'}`` for training and ``test`` for
        evaluation. If False, use a random stratified 80/20 split. If None (default),
        auto-detect: official split when ``split`` column has train/dev and test rows.

    Returns
    -------
    X_train : array-like
        Standardized training feature matrix.
    X_test : array-like
        Standardized test feature matrix.
    y_train : array-like
        Training labels indicating dropout status.
    y_test : array-like
        Test labels indicating dropout status.
    scaler : StandardScaler
        Fitted scaler object used to transform features.
    """
    if "dropout" not in df.columns:
        raise ValueError("DataFrame must contain a 'dropout' column as the target.")

    cols = list(feature_columns) if feature_columns is not None else _resolve_feature_columns(df)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in DataFrame: {missing}")

    X = df[cols]
    y = df["dropout"]

    if use_official_split is None:
        use_official_split = _should_use_official_split(df)

    if use_official_split:
        if "split" not in df.columns:
            raise ValueError("use_official_split=True requires a 'split' column on the DataFrame.")
        train_mask = df["split"].isin(["train", "dev"])
        test_mask = df["split"].eq("test")
        if not train_mask.any():
            raise ValueError("Official split requested but no rows with split in {'train','dev'}.")
        if not test_mask.any():
            raise ValueError("Official split requested but no rows with split == 'test'.")
        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(X_train: Any, y_train: Any) -> Tuple[Any, Any]:
    """
    Apply SMOTE to address class imbalance in psychotherapy dropout labels.

    In many clinical datasets, dropout events (patients who prematurely end
    treatment) are less frequent than non-dropouts, which can bias models
    toward predicting the majority class. SMOTE (Synthetic Minority Over-
    sampling Technique) generates synthetic examples of the minority class
    in the training data only, improving the model's ability to detect
    high-risk patients without contaminating the test set.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix before resampling.
    y_train : array-like
        Training labels indicating dropout status before resampling.

    Returns
    -------
    X_train_resampled : array-like
        Training features after SMOTE oversampling.
    y_train_resampled : array-like
        Training labels after SMOTE oversampling.

    Notes
    -----
    - SMOTE is applied strictly to the training set; the test set should
      remain untouched to preserve a realistic evaluation.
    """
    print("Class distribution before SMOTE:")
    print(pd.Series(y_train).value_counts(normalize=False).rename("count"))
    print(
        pd.Series(y_train)
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename("pct_%")
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts(normalize=False).rename("count"))
    print(
        pd.Series(y_train_resampled)
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename("pct_%")
    )

    return X_train_resampled, y_train_resampled


def train_model(X_train: Any, y_train: Any, model_type: str) -> Any:
    """
    Train a psychotherapy dropout prediction model of the requested type.

    This function supports three clinically relevant model families:
    - 'logistic_regression': interpretable linear model often used in clinical research.
    - 'random_forest': non-linear ensemble capturing complex interactions between
       symptoms, engagement patterns, and dropout.
    - 'xgboost': gradient boosting model that can capture subtle predictive signals
       in high-dimensional feature spaces.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix (optionally SMOTE-resampled).
    y_train : array-like
        Training labels indicating dropout status.
    model_type : str
        One of {'logistic_regression', 'random_forest', 'xgboost'}.

    Returns
    -------
    model : Any
        Trained model instance ready for evaluation and inference.
    """
    model_type = model_type.lower()

    if model_type == "logistic_regression":
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )
    else:
        raise ValueError(
            "Unsupported model_type. Choose from "
            "'logistic_regression', 'random_forest', or 'xgboost'."
        )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate a trained dropout prediction model using clinically meaningful metrics.

    Rather than overall accuracy (which can be misleading in imbalanced
    clinical datasets), this function focuses on:
    - AUC-ROC: ability to discriminate between dropout and non-dropout patients.
    - F1 score: balance between precision and recall for the dropout class.
    - Precision: proportion of predicted dropouts who actually drop out
      (helps avoid unnecessary interventions).
    - Recall: proportion of true dropouts correctly identified
      (sensitivity to high-risk patients).

    Parameters
    ----------
    model : Any
        Trained classifier with a predict and predict_proba (or decision_function) method.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True dropout labels for the test set.

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        - 'auc_roc'
        - 'f1'
        - 'precision'
        - 'recall'
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler

        decision_scores = model.decision_function(X_test).reshape(-1, 1)
        y_proba = MinMaxScaler().fit_transform(decision_scores).ravel()
    else:
        raise ValueError(
            "Model must support either predict_proba or decision_function for AUC-ROC."
        )

    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    metrics: Dict[str, float] = {
        "auc_roc": float(auc_roc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }

    return metrics


def run_all_models(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    use_official_split: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run the full psychotherapy dropout prediction pipeline for three model families.

    This function:
    1. Prepares the data (train/test split and scaling).
    2. Applies SMOTE to address dropout class imbalance in the training set.
    3. Trains three models in sequence:
       - Logistic Regression
       - Random Forest
       - XGBoost
    4. Evaluates each model using AUC-ROC, F1, Precision, and Recall.
    5. Prints a side-by-side comparison of model performance, helping clinicians
       and researchers choose a model that best balances sensitivity to dropout
       with precision and overall discrimination.

    Parameters
    ----------
    df : pd.DataFrame
        Full psychotherapy dataset with 'dropout' as the target and all
        relevant clinical features already prepared.
    feature_columns : sequence of str, optional
        Passed through to ``prepare_data``.
    use_official_split : bool, optional
        Passed through to ``prepare_data`` (DAIC-WOZ official train+dev vs test).

    Returns
    -------
    models : dict
        Dictionary mapping model names to their trained instances:
        - 'logistic_regression'
        - 'random_forest'
        - 'xgboost'
    """
    X_train, X_test, y_train, y_test, _ = prepare_data(
        df,
        feature_columns=feature_columns,
        use_official_split=use_official_split,
    )
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    model_types = ["logistic_regression", "random_forest", "xgboost"]
    models: Dict[str, Any] = {}
    results: Dict[str, Dict[str, float]] = {}

    for m_type in model_types:
        print(f"\nTraining model: {m_type}")
        model = train_model(X_train_res, y_train_res, m_type)
        models[m_type] = model
        metrics = evaluate_model(model, X_test, y_test)
        results[m_type] = metrics

    print("\nModel performance comparison (AUC-ROC, F1, Precision, Recall):")
    header = (
        f"{'Model':<20} {'AUC-ROC':>10} {'F1':>10} {'Precision':>12} {'Recall':>10}"
    )
    print(header)
    print("-" * len(header))

    for m_type in model_types:
        m_res = results[m_type]
        print(
            f"{m_type:<20} "
            f"{m_res['auc_roc']:>10.4f} "
            f"{m_res['f1']:>10.4f} "
            f"{m_res['precision']:>12.4f} "
            f"{m_res['recall']:>10.4f}"
        )

    return models
