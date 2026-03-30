# Churn Prediction — Project Documentation

This repository contains a small end-to-end **Telco customer churn** modeling workflow: exploratory analysis (including saved figures), preprocessing, train/test split, two classifiers (**Logistic Regression** and **XGBoost**), evaluation metrics, and **MLflow** experiment tracking. It is suitable for a data science portfolio demo.

---

## What’s in this repo

| Item | Purpose |
|------|---------|
| `churn.csv` | Telco churn dataset (one row per customer; target column `Churn`: Yes/No). |
| `train_churn_models.py` | Main script: load → explore → preprocess → split → train → evaluate → log to MLflow → print comparison. |
| `eda_telco_churn.py` | EDA script: builds three matplotlib/seaborn figures and saves them as PNGs under `figures/`. |
| `figures/` | Created when you run the EDA script — contains the churn visualizations (optional to commit). |
| `mlruns/` | Created after the first training run — local MLflow tracking store (gitignored if you add it to `.gitignore`; optional to commit). |

Scripts resolve paths relative to **their own location**, so they find `churn.csv` beside the script as long as you keep that layout.

---

## Prerequisites

- **Python 3.9+** (3.9 is fine; newer versions work if dependencies install cleanly).
- Python packages:

  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `mlflow`
  - `matplotlib` and `seaborn` (for `eda_telco_churn.py`)

Install them (ideally in a virtual environment):

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pandas scikit-learn xgboost mlflow matplotlib seaborn
```

---

## How the script works

### 1. Load and explore

- Reads `churn.csv` with pandas.
- Prints **shape** (rows × columns), **churn value counts** and **churn rate** (proportion of `Churn == Yes`), **missing values** per column on the raw read, and a short **preview** of the first rows.
- Drops **`customerID`** (identifier, not for modeling).
- Converts **`TotalCharges`** to numeric with `errors="coerce"` so empty strings become missing values to be handled in preprocessing.

### 2. Preprocess

- **Target:** `Churn` encoded as binary (1 = Yes, 0 = No).
- **Numeric features** (inferred from dtypes): median imputation for missing values, then **`StandardScaler`**.
- **Categorical features** (everything else after dropping ID/target): most-frequent imputation, then **`OneHotEncoder`** with `handle_unknown="ignore"` so unseen categories at test time do not break the pipeline.
- Preprocessing is wrapped in scikit-learn **`ColumnTransformer`** and combined with each estimator inside a **`Pipeline`**, so statistics are learned **only on the training split** (no leakage from the test set).

### 3. Train / test split

- **80% train / 20% test**, `random_state=42`.
- **Stratified** on the churn label so train and test churn rates match the full dataset.

### 4. Models

Two separate pipelines share the same preprocessing *design* (each pipeline fits its own preprocessor):

1. **Logistic Regression** — `class_weight="balanced"`, `solver="saga"` (stable with dense one-hot features), `max_iter=5000`.
2. **XGBoost** — `XGBClassifier` with logged hyperparameters (e.g. `n_estimators`, `max_depth`, `learning_rate`, `eval_metric="logloss"`).

### 5. Metrics

On the **held-out test set**, using predicted **probabilities** for ROC-AUC and a **0.5** threshold for discrete predictions:

- **ROC-AUC**
- **F1 score**
- **Precision**
- **Recall**  
- **Accuracy** (included in the printed comparison table)

### 6. MLflow

- **Tracking URI:** file-based store under `./mlruns` (next to the script).
- **Experiment name:** `telco_churn_portfolio`.
- **Two runs:** `logistic_regression` and `xgboost`.
- Each run logs **hyperparameters**, **metrics**, and the **full sklearn `Pipeline`** as a model artifact (`name="model"`), with an **`input_example`** for signature inference.

### 7. Comparison

After both runs finish, the script prints a **side-by-side table** of test metrics and notes which model has higher ROC-AUC on that split.

---

## EDA visualizations

The script **`eda_telco_churn.py`** reads `churn.csv` from the project folder and writes **three PNG files** into **`figures/`** (the directory is created automatically). It uses **matplotlib** and **seaborn** with a simple **whitegrid** theme.

### Output files

| File | Description |
|------|-------------|
| `figures/churn_rate_by_contract.png` | **Churn rate by contract type** — proportion of customers with `Churn == Yes` within each `Contract` category, bars sorted from highest to lowest rate, with the rate labeled on each bar. |
| `figures/churn_rate_by_tenure_bucket.png` | **Churn rate by tenure bucket** — same idea, but `tenure` (months) is binned into **0–12**, **13–24**, **25–48**, and **49–72** months. |
| `figures/roc_lr_vs_xgboost.png` | **ROC curve comparison** — **Logistic Regression** vs **XGBoost** on the **held-out test set**, using the same **80/20 stratified** split (`random_state=42`) and the same preprocessing idea as `train_churn_models.py` (median/mode imputation, `StandardScaler` on numerics, one-hot encoding on categoricals). The legend includes **AUC** for each model and a dashed **chance** diagonal (AUC = 0.50). |

The script filters a few noisy **RuntimeWarning** messages from `sklearn.utils.extmath` so the console output stays readable; they do not affect the saved figures.

### How to run the EDA script

From the project root, with your virtual environment activated:

```bash
cd "/path/to/Churn Prediction"
source venv/bin/activate
python3 eda_telco_churn.py
```

If XGBoost fails to import on macOS, follow the **OpenMP** instructions in the **Environment note** section at the bottom of this document.

---

## How to run (end to end)

1. **Clone or copy** the project and `cd` into the project root (the folder that contains `churn.csv` and `train_churn_models.py`).

2. **Create and activate a virtual environment** (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies** (if not already):

```bash
pip install pandas scikit-learn xgboost mlflow matplotlib seaborn
```

4. **Run the training script:**

```bash
python3 train_churn_models.py
```

You should see exploration output, split sizes, MLflow tracking messages, and the final model comparison table.

5. **(Optional) Generate EDA figures:**

```bash
python3 eda_telco_churn.py
```

Check the `figures/` folder for the three PNGs described in the **EDA visualizations** section above.

### View experiments in the MLflow UI

From the same project directory (with the venv activated):

```bash
mlflow ui
```

Open the URL shown in the terminal (usually `http://127.0.0.1:5000`). You will see the `telco_churn_portfolio` experiment and the two runs with parameters, metrics, and saved model artifacts.

---

## Environment note: macOS, XGBoost, and OpenMP

On **macOS**, the XGBoost Python wheel links against **OpenMP** (`libomp`). If importing or running XGBoost fails with an error about **`libxgboost.dylib`** or **`libomp.dylib`** not loading, install the OpenMP runtime with **Homebrew**:

```bash
brew install libomp
```

Then try running the script again in the same terminal. This is a common one-time setup step on Apple Silicon and Intel Macs when the library is not already present on the system.

---

## Quick troubleshooting

| Issue | What to try |
|-------|----------------|
| `python: command not found` | Use `python3`, or on macOS ensure Python 3 is installed (e.g. `python3 --version`). |
| `FileNotFoundError` for `churn.csv` | Run from the project root and keep `churn.csv` next to the scripts (`train_churn_models.py`, `eda_telco_churn.py`). |
| MLflow UI empty | Run `train_churn_models.py` once first; confirm `./mlruns` exists and you started `mlflow ui` from the project directory. |
| EDA script errors on `matplotlib` / `seaborn` | Install visualization dependencies: `pip install matplotlib seaborn`. |

---

*Last updated to include `eda_telco_churn.py` and `figures/`, plus the `train_churn_models.py` workflow (experiment `telco_churn_portfolio`, stratified 80/20 split, Logistic Regression + XGBoost, MLflow file store under `./mlruns`).*
