#!/usr/bin/env python3
"""
Telco churn prediction: exploratory analysis, preprocessing, and two models
(Logistic Regression + XGBoost) with MLflow tracking.

Run from the project folder:
    python train_churn_models.py
Requires: pandas, scikit-learn, xgboost, mlflow
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
# Resolve CSV next to this script so it works regardless of cwd
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "churn.csv"

TARGET_COL = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# MLflow experiment stores both runs under one experiment name
EXPERIMENT_NAME = "telco_churn_portfolio"
MLFLOW_TRACKING_URI = "file:" + str(SCRIPT_DIR / "mlruns")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the Telco churn CSV into a DataFrame."""
    return pd.read_csv(path)


def explore_data(df: pd.DataFrame) -> None:
    """
    Print dataset shape, missing values, class balance (churn rate),
    and a small preview for portfolio documentation.
    """
    print("=" * 60)
    print("1. DATA EXPLORATION")
    print("=" * 60)

    # Rows × columns
    print(f"\nShape (rows, columns): {df.shape}")

    # Target distribution — churn rate
    if TARGET_COL in df.columns:
        churn_counts = df[TARGET_COL].value_counts()
        churn_rate = (df[TARGET_COL] == "Yes").mean()
        print(f"\nChurn value counts:\n{churn_counts}")
        print(f"\nChurn rate (proportion Yes): {churn_rate:.4f}")

    # Missing values per column (including empty strings if any)
    nulls = df.isna().sum()
    # Telco dataset often has TotalCharges as object; empty cells become NaN after numeric parse
    print("\nMissing values per column (raw read):")
    print(nulls[nulls > 0].to_string() if nulls.any() else "  None")

    print("\nFirst 3 rows (preview):")
    print(df.head(3).to_string())
    print()


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Drop ID column, coerce types, encode target as 0/1.

    Returns
    -------
    X : feature DataFrame
    y : binary numpy array (1 = churn)
    """
    df = df.copy()

    # Drop identifier — not predictive and would leak identity
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # TotalCharges is sometimes blank for new accounts; coerce then impute later in pipeline
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = (df[TARGET_COL] == "Yes").astype(int).values
    X = df.drop(columns=[TARGET_COL])
    return X, y


def make_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer: median/mode imputation, OHE for categoricals,
    StandardScaler for numeric features.

    Fitted inside each sklearn Pipeline so train statistics never leak from test.
    """
    # Numeric columns (after dropping ID and target)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    # Everything else is treated as categorical (strings / objects)
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Numeric branch: fill missing (e.g. TotalCharges), then z-score scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical branch: most frequent for missing, one-hot with unknown handling
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics at a fixed threshold on predicted probabilities.

    Returns a dict suitable for MLflow logging and console comparison.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def print_comparison(metrics_lr: dict, metrics_xgb: dict) -> None:
    """Print a side-by-side table of the two models."""
    print("\n" + "=" * 60)
    print("7. MODEL COMPARISON (test set)")
    print("=" * 60)
    rows = [
        ("ROC-AUC", "Higher is better", metrics_lr["roc_auc"], metrics_xgb["roc_auc"]),
        ("F1 score", "Balance of precision & recall", metrics_lr["f1"], metrics_xgb["f1"]),
        ("Precision", "Of predicted churn, how many really churned", metrics_lr["precision"], metrics_xgb["precision"]),
        ("Recall", "Of actual churners, how many we catch", metrics_lr["recall"], metrics_xgb["recall"]),
        ("Accuracy", "Overall correct rate", metrics_lr["accuracy"], metrics_xgb["accuracy"]),
    ]
    w = 14
    print(f"\n{'Metric':<{w}}{'Logistic Regression':>22}{'XGBoost':>22}")
    print("-" * (w + 22 + 22))
    for name, _, v_lr, v_xgb in rows:
        print(f"{name:<{w}}{v_lr:>22.4f}{v_xgb:>22.4f}")
    print()
    best_auc = "Logistic Regression" if metrics_lr["roc_auc"] >= metrics_xgb["roc_auc"] else "XGBoost"
    print(f"Higher ROC-AUC on this split: {best_auc}")
    print()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- 1. Load & explore ---
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = load_raw_data(DATA_PATH)
    explore_data(df)

    X, y = build_features_and_target(df)

    # --- 2. Preprocessing blueprint (fit happens inside each model pipeline) ---
    preprocessor = make_preprocessing_pipeline(X)

    # --- 3. Train / test split: 80/20, stratify on churn to preserve class ratio ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("=" * 60)
    print("2–3. PREPROCESSING & SPLIT")
    print("=" * 60)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train churn rate: {y_train.mean():.4f}, Test churn rate: {y_test.mean():.4f}\n")

    # --- 4. Model definitions ---
    # saga handles dense one-hot features more stably than lbfgs for this pipeline
    lr_params = {
        "C": 1.0,
        "max_iter": 5000,
        "solver": "saga",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    pipe_lr = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**lr_params)),
        ]
    )

    xgb_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
    }
    # Second preprocessor instance — same config, independent fit per pipeline
    preprocessor_xgb = make_preprocessing_pipeline(X)
    pipe_xgb = Pipeline(
        steps=[
            ("preprocessor", preprocessor_xgb),
            ("classifier", XGBClassifier(**xgb_params)),
        ]
    )

    # --- MLflow: local file store, one experiment, two runs ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    metrics_lr: dict | None = None
    metrics_xgb: dict | None = None

    # ----- Run 1: Logistic Regression -----
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_params({f"lr__{k}": v for k, v in lr_params.items()})
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        pipe_lr.fit(X_train, y_train)
        proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
        metrics_lr = evaluate_model(y_test, proba_lr)

        for name, value in metrics_lr.items():
            mlflow.log_metric(name, value)

        # Full sklearn Pipeline (preprocessing + model) as artifact
        mlflow.sklearn.log_model(
            pipe_lr,
            name="model",
            input_example=X_train.head(1),
        )

    # ----- Run 2: XGBoost -----
    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({f"xgb__{k}": v for k, v in xgb_params.items()})
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        pipe_xgb.fit(X_train, y_train)
        proba_xgb = pipe_xgb.predict_proba(X_test)[:, 1]
        metrics_xgb = evaluate_model(y_test, proba_xgb)

        for name, value in metrics_xgb.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(
            pipe_xgb,
            name="model",
            input_example=X_train.head(1),
        )

    assert metrics_lr is not None and metrics_xgb is not None

    # --- 5–6. Metrics already computed & logged; print human-readable comparison ---
    print("=" * 60)
    print("4–6. TRAINING COMPLETE — metrics logged to MLflow")
    print("=" * 60)
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("View UI:  mlflow ui  (from project dir, same venv)")
    print()

    print_comparison(metrics_lr, metrics_xgb)


if __name__ == "__main__":
    main()
