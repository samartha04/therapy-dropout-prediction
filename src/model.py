import os
import pickle
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

    Splits data into train/test sets and applies StandardScaler.
    Scaler is saved to models/scaler.pkl for use during inference.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with dropout target column.
    feature_columns : sequence of str, optional
        Feature names to use. Auto-detected if None.
    use_official_split : bool, optional
        Use official DAIC-WOZ split if available.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
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
        train_mask = df["split"].isin(["train", "dev"])
        test_mask = df["split"].eq("test")
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

    # Save scaler for inference
    os.makedirs("models", exist_ok=True)
    scaler_path = os.path.join("models", "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(X_train: Any, y_train: Any) -> Tuple[Any, Any]:
    """
    Apply SMOTE to address class imbalance in psychotherapy dropout labels.

    SMOTE is applied strictly to the training set only.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix before resampling.
    y_train : array-like
        Training labels before resampling.

    Returns
    -------
    X_train_resampled, y_train_resampled
    """
    print("Class distribution before SMOTE:")
    print(pd.Series(y_train).value_counts(normalize=False).rename("count"))

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts(normalize=False).rename("count"))

    return X_train_resampled, y_train_resampled


def train_model(X_train: Any, y_train: Any, model_type: str) -> Any:
    """
    Train a psychotherapy dropout prediction model.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training labels.
    model_type : str
        One of logistic_regression, random_forest, xgboost.

    Returns
    -------
    model : trained model instance
    """
    model_type = model_type.lower()

    if model_type == "logistic_regression":
        model = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            C=0.1,
            random_state=42,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1.0,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )
    else:
        raise ValueError(
            "Unsupported model_type. Choose from "
            "logistic_regression, random_forest, or xgboost."
        )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """
    Evaluate a trained dropout prediction model.

    Uses AUC-ROC, F1, Precision, Recall — never raw accuracy.

    Parameters
    ----------
    model : Any
        Trained classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True dropout labels.

    Returns
    -------
    metrics : dict with auc_roc, f1, precision, recall
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must support predict_proba.")

    y_pred = model.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "auc_roc": float(auc_roc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def run_all_models(
    df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    use_official_split: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run full dropout prediction pipeline for all three models.

    Trains Logistic Regression, Random Forest, and XGBoost.
    Prints side-by-side comparison of AUC-ROC, F1, Precision, Recall.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with dropout target.
    feature_columns : sequence of str, optional
        Feature names to use.
    use_official_split : bool, optional
        Use official DAIC-WOZ split if available.

    Returns
    -------
    models : dict of trained model instances
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

    print("\nModel performance comparison:")
    header = f"{'Model':<20} {'AUC-ROC':>10} {'F1':>10} {'Precision':>12} {'Recall':>10}"
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