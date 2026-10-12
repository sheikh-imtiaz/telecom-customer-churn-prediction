# Telecommunications Customer Churn Prediction

A reproducible, portfolio-ready machine learning project to explore **telecommunications customer churn**, identify associated customer segments, and predict which customers may leave a provider.

## Project overview

The project investigates demographic, subscription, billing, and service factors associated with churn. It combines exploratory data analysis, a chi-square hypothesis test, and classification models. The notebook compares **Logistic Regression**, **Random Forest**, and **Gradient Boosting**, tunes the strongest cross-validated baseline, evaluates an untouched holdout set, and exports a reusable prediction pipeline.

> **Important:** Results depend on running the notebook with the actual data. This repository does not claim precomputed accuracy or ROC-AUC results.

## Business questions

1. Which demographic segments have higher observed churn rates?
2. How is customer tenure associated with churn?
3. Which internet and support services are associated with churn?
4. How do contract types and payment methods relate to churn?
5. How do monthly and total charges differ between retained and churned customers?

**Hypothesis:** The null hypothesis is that internet service type and customer churn are independent. The notebook uses a chi-square test of independence at a 5% significance level. Statistical association does not imply causation.

## Repository structure

```text
telecom-customer-churn/
├── telecom_customer_churn_optimized.ipynb
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                  # Add local CSV files; excluded from Git
├── outputs/               # Generated charts, scores, predictions; excluded from Git
└── models/                # Exported trained pipeline; excluded from Git
```

## Data sources

The original notebook uses three batches:

| Dataset | Purpose | Source |
| --- | --- | --- |
| `LP2_Telco_churn_first_3000` | Labeled training records | SQL Server table `dbo.LP2_Telco_churn_first_3000` **or** `data/LP2_Telco-churn-first-3000.csv` |
| `LP2_Telco-churn-second-2000 (1).csv` | Additional labeled training records | `data/` |
| `Telco-churn-last-2000.csv` | Final unseen prediction records | `data/` |

The first two datasets are concatenated and deduplicated before model development. The third is held apart for prediction and **is not used for evaluation**. Filenames can be changed in the configuration cell. Keep customer records private and verify redistribution rights before publishing any data.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `telecom_customer_churn_optimized.ipynb` and run the cells from top to bottom.

### SQL Server option

If the first-batch CSV is not present, the notebook can query the original SQL Server table. Install the optional dependencies:

```bash
pip install pyodbc python-dotenv
```

Install a compatible Microsoft SQL Server ODBC driver, copy `.env.example` to `.env`, and set the values. The SQL connection uses encryption and requires a trusted server certificate. **Never commit `.env` or database passwords.**

## Analysis and modeling workflow

1. Load and combine labeled customer records.
2. Standardize data types and handle missing target labels and duplicates.
3. Explore churn distribution, customer tenure, demographic and service segments, and billing.
4. Test the relationship between internet service and churn with chi-square.
5. Make a stratified 80/20 train/holdout split.
6. Build a `ColumnTransformer` that imputes, scales, and one-hot encodes **inside** each training fold.
7. Compare three classifiers using stratified five-fold CV (ROC-AUC, average precision, F1).
8. Tune the leading model using `GridSearchCV` on training data only.
9. Evaluate once on the holdout set (ROC-AUC, average precision, accuracy, precision, recall, F1).
10. Save the fitted pipeline and generate probabilities and predicted labels for the unseen customer batch.

**Leakage prevention:** no global preprocessing or SMOTE is fitted before the split. The notebook does not use the final prediction dataset for tuning. The default decision threshold is 0.50; business-specific thresholds require separate validation.

## Outputs

After a successful run:

- `outputs/churn_distribution.png`
- `outputs/model_cv_comparison.csv`
- `outputs/holdout_model_evaluation.png`
- `outputs/holdout_metrics.csv`
- `outputs/unseen_customer_churn_predictions.csv`
- `models/telecom_customer_churn_pipeline.joblib`

Outputs may contain sensitive customer data. They are ignored by Git by default. Load serialized models only from trusted sources.

## Key libraries

Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, joblib, Jupyter.

## Limitations and responsible use

This is an educational and portfolio-oriented predictive analysis. It does not demonstrate causality, guarantee generalization to future time periods, or establish probability calibration. Evaluate temporal drift, group-level performance, data governance, and the costs of false positives/negatives before operational use. Never publish customer identifiers or raw customer-level records without permission.

## Future improvements

- Probability calibration and business-cost-based threshold selection.
- Time-aware validation where timestamps exist.
- Feature importance or SHAP explanations for the selected model.
- Model monitoring and drift detection.
- Automated data-quality and pipeline tests.

## License

No license is granted by default. Add a license only if you own the relevant code and have the rights to distribute it and any included data.
