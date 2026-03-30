#!/usr/bin/env python3
"""
Exploratory visualizations for the Telco churn dataset.

Produces three PNG figures:
  1. Churn rate by contract type
  2. Churn rate by tenure bucket
  3. ROC curves: Logistic Regression vs XGBoost (same train/test split as training script)

Run from project root:
    python3 eda_telco_churn.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# -----------------------------------------------------------------------------
# Paths (same layout as train_churn_models.py)
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "churn.csv"
FIGURES_DIR = SCRIPT_DIR / "figures"

TARGET_COL = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_churn_rate_by_contract(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: churn rate (proportion Yes) within each contract type."""
    work = df.copy()
    work["_churn"] = (work[TARGET_COL] == "Yes").astype(float)

    rates = (
        work.groupby("Contract", observed=True)["_churn"]
        .mean()
        .reset_index()
        .rename(columns={"_churn": "churn_rate"})
        .sort_values("churn_rate", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=rates,
        x="Contract",
        y="churn_rate",
        hue="Contract",
        palette="Blues_d",
        legend=False,
    )
    ax.set_title("Churn rate by contract type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, max(rates["churn_rate"].max() * 1.15, 0.05))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_churn_rate_by_tenure_bucket(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: churn rate within tenure buckets (months)."""
    work = df.copy()
    work["_churn"] = (work[TARGET_COL] == "Yes").astype(float)

    # Business-friendly buckets (tenure is in months; max in dataset is 72)
    tenure_bins = [-0.01, 12, 24, 48, 72]
    tenure_labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"]
    work["tenure_bucket"] = pd.cut(
        work["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
    )

    rates = (
        work.groupby("tenure_bucket", observed=True)["_churn"]
        .mean()
        .reset_index()
        .rename(columns={"_churn": "churn_rate"})
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=rates,
        x="tenure_bucket",
        y="churn_rate",
        hue="tenure_bucket",
        palette="YlOrRd_d",
        legend=False,
    )
    ax.set_title("Churn rate by tenure bucket")
    ax.set_xlabel("Tenure bucket")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, max(rates["churn_rate"].max() * 1.15, 0.05))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df[TARGET_COL] == "Yes").astype(int).values
    X = df.drop(columns=[TARGET_COL])
    return X, y


def make_preprocessing_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def plot_roc_lr_vs_xgboost(X_train, X_test, y_train, y_test, out_path: Path) -> None:
    """
    Train LR and XGBoost with the same preprocessing as train_churn_models.py,
    then overlay ROC curves on the test set.
    """
    preprocessor_lr = make_preprocessing_pipeline(X_train)
    pipe_lr = Pipeline(
        steps=[
            ("preprocessor", preprocessor_lr),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    preprocessor_xgb = make_preprocessing_pipeline(X_train)
    pipe_xgb = Pipeline(
        steps=[
            ("preprocessor", preprocessor_xgb),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipe_lr.fit(X_train, y_train)
    pipe_xgb.fit(X_train, y_train)

    proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
    proba_xgb = pipe_xgb.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        proba_lr,
        name="Logistic Regression",
        ax=ax,
        color="#1f77b4",
    )
    RocCurveDisplay.from_predictions(
        y_test,
        proba_xgb,
        name="XGBoost",
        ax=ax,
        color="#d62728",
    )
    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance (AUC = 0.50)")

    fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)
    auc_lr = auc(fpr_lr, tpr_lr)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    ax.set_title("ROC curve — Logistic Regression vs XGBoost (test set)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    # Legend with AUC in title area / subtitle via text box
    handles, labels = ax.get_legend_handles_labels()
    # Replace default names with AUC annotations
    legend_labels = [
        f"Logistic Regression (AUC = {auc_lr:.3f})",
        f"XGBoost (AUC = {auc_xgb:.3f})",
        "Chance (AUC = 0.50)",
    ]
    ax.legend(handles=handles, labels=legend_labels, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    # Benign matmul noise from some sklearn paths (e.g. linear model / metrics)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.facecolor"] = "white"

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(DATA_PATH)

    plot_churn_rate_by_contract(df, FIGURES_DIR / "churn_rate_by_contract.png")
    plot_churn_rate_by_tenure_bucket(df, FIGURES_DIR / "churn_rate_by_tenure_bucket.png")

    X, y = build_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    plot_roc_lr_vs_xgboost(
        X_train,
        X_test,
        y_train,
        y_test,
        FIGURES_DIR / "roc_lr_vs_xgboost.png",
    )

    print(f"\nAll figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
