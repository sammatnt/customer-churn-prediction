# Telco Customer Churn Prediction

End-to-end **binary classification** project predicting whether a telecom customer will churn. Built as a **portfolio-ready** workflow: exploratory analysis, reproducible preprocessing, two baseline models, evaluation, and **MLflow** experiment tracking.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Seaborn-7C4D79?style=for-the-badge" alt="Seaborn" />
</p>

---

## Business problem

**Churn**—customers canceling service—directly reduces recurring revenue and increases acquisition costs. If the business can **identify at-risk customers early**, teams can target **retention offers**, **support outreach**, or **plan upgrades** before cancellation.

This project frames churn as a **supervised learning** problem: learn patterns from historical customer attributes and produce a **ranked risk score** (probability of churn) suitable for prioritization, plus interpretable baselines for stakeholder discussion.

---

## Dataset

| Property | Detail |
|----------|--------|
| **Source** | Public **Telco Customer Churn** tabular dataset (IBM sample / Kaggle-style variant; one row per customer). |
| **Size** | ~7,043 customers × **21** columns (including ID and target). |
| **Target** | `Churn`: **Yes** / **No** (~26.5% **Yes** in the provided file—imbalanced but workable with stratified splits and class weights). |
| **Features** | Demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account tenure and charges (`tenure`, `MonthlyCharges`, `TotalCharges`), services (`PhoneService`, `InternetService`, add-ons), and contract/billing (`Contract`, `PaperlessBilling`, `PaymentMethod`). |

The raw file **`churn.csv`** ships in the repo root. `customerID` is excluded from modeling; `TotalCharges` is parsed as numeric with invalid/blank values imputed in the pipeline.

---

## Approach

| Stage | What we do |
|-------|------------|
| **EDA** | Summary stats, churn rate views by **contract** and **tenure** buckets, and a **ROC** comparison plot (`eda_telco_churn.py` → PNGs in `figures/`). |
| **Preprocessing** | **Median** imputation + **StandardScaler** on numeric columns; **most-frequent** imputation + **one-hot encoding** on categoricals (`handle_unknown="ignore"`). All steps live in **`sklearn` `Pipeline`s** so test data never leaks statistics from the train set. |
| **Model training** | **80/20** train–test split, **stratified** on churn (`random_state=42`). Two models: **Logistic Regression** (balanced class weights, `saga` solver) and **XGBoost** gradient boosted trees. |
| **Evaluation** | **ROC-AUC**, **F1**, precision, recall, and accuracy on the held-out test set; full runs logged in **MLflow** (params, metrics, serialized pipelines). |

For implementation details, see **[DOCUMENTATION.md](DOCUMENTATION.md)**.

---

## Results

**Test set** (20% holdout, stratified). Metrics use **predicted probabilities** for ROC-AUC and a **0.5** decision threshold for F1 / precision / recall.

| Model | ROC-AUC | F1 |
|-------|---------|-----|
| **Logistic Regression** | **0.842** | **0.614** |
| **XGBoost** | 0.834 | 0.573 |

On this split, **logistic regression** edges out XGBoost on **ROC-AUC** and **F1**; XGBoost shows **higher precision** and **lower recall** at 0.5 threshold—useful context if the business prioritizes fewer false alarms vs catching every churner. In production you would tune the threshold using cost-sensitive criteria and validate with cross-validation or a separate validation set.

---

## Key findings (what drives churn)

Patterns align with typical telecom behavior and the project’s EDA visuals:

1. **Contract type** — **Month-to-month** customers churn at a much higher rate than those on **one-year** or **two-year** contracts; commitment length is one of the strongest levers.
2. **Tenure** — Churn is **concentrated among newer customers** (e.g. first **12 months**); longer tenure strongly associates with **retention**.
3. **Services & billing** — In this dataset, **fiber** internet and certain **payment methods** (e.g. electronic check) often co-occur with higher churn in group summaries—worth pairing with model coefficients or SHAP in a follow-up for causal language.
4. **Charges** — Higher **monthly charges** frequently appear among churners; combined with contract and tenure, this supports **value / pricing sensitivity** narratives for retention campaigns.

*These are descriptive and model-agnostic insights; for product decisions, validate with domain experts and experiments.*

---

## Repository layout

```
Churn Prediction/
├── churn.csv                 # Dataset
├── train_churn_models.py     # Training, metrics, MLflow logging
├── eda_telco_churn.py        # EDA figures (PNG)
├── figures/                  # Generated plots (after running EDA)
├── mlruns/                   # MLflow local store (after training)
├── DOCUMENTATION.md          # Detailed technical reference
└── README.md
```

---

## Getting started (local)

**Prerequisites:** Python **3.9+** and `pip`.

```bash
# Clone and enter the project
git clone <your-repo-url>
cd "Churn Prediction"

# Virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Dependencies
pip install pandas scikit-learn xgboost mlflow matplotlib seaborn

# Train models + log to MLflow
python3 train_churn_models.py

# Optional: EDA figures → figures/*.png
python3 eda_telco_churn.py

# Optional: browse experiments
mlflow ui
```

Open the URL printed in the terminal (often `http://127.0.0.1:5000`) to inspect experiment **`telco_churn_portfolio`**.

### macOS + XGBoost

If XGBoost fails to load (`libomp` / `libxgboost`), install OpenMP once:

```bash
brew install libomp
```

---

## Sample outputs

After running **`eda_telco_churn.py`**, you can embed figures in your portfolio README (commit `figures/` or host elsewhere):

| Figure | Path |
|--------|------|
| Churn by contract | `figures/churn_rate_by_contract.png` |
| Churn by tenure | `figures/churn_rate_by_tenure_bucket.png` |
| ROC comparison | `figures/roc_lr_vs_xgboost.png` |

---

## Author

Portfolio project — **Telco customer churn** classification with **scikit-learn**, **XGBoost**, and **MLflow**.
