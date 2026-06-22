# 🏦 Loan Approval Predictor

**CodTech IT Solutions — ML Internship**

| Field | Details |
|---|---|
| **Intern ID** | CITS5203 |
| **Full Name** | Anagh Pandey |
| **No. of Weeks** | 4 |
| **Project Name** | Loan Approval Predictor |
| **Project Scope** | Machine Learning |

---

## 📌 Project Overview

A machine learning project that predicts loan approval decisions based on applicant demographics, financial profile, and loan details. Three classifiers are trained and compared using accuracy, precision, recall, F1, ROC-AUC, and cross-validation.

---

## 🗂️ Project Structure

```
loan-approval-predictor/
├── data/
│   └── loan_data.csv                 # Generated dataset (1000 applicants)
├── outputs/                          # All output charts
│   ├── 01_approval_rate.png
│   ├── 02_approval_by_employment.png
│   ├── 03_credit_score_dist.png
│   ├── 04_income_vs_loan.png
│   ├── 05_dti_boxplot.png
│   ├── 06_approval_by_purpose.png
│   ├── 07_approval_by_education.png
│   ├── 08_correlation_heatmap.png
│   ├── 09_confusion_matrix.png
│   ├── 10_roc_curves.png
│   ├── 11_feature_importance.png
│   ├── 12_model_comparison.png
│   └── 13_cross_validation.png
├── generate_data.py                  # Generates loan applicant dataset
├── visualize.py                      # EDA visualizations
├── model.py                          # Model training + evaluation
├── main.py                           # Run everything with one command
└── requirements.txt
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

Or step by step:
```bash
python generate_data.py   # Generate dataset
python visualize.py       # EDA charts
python model.py           # Train & evaluate models
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `age` | Applicant age |
| `gender` | Male / Female |
| `education` | Education level |
| `employment_type` | Salaried / Self-Employed / Business / Unemployed |
| `annual_income` | Annual income (₹) |
| `loan_amount` | Requested loan amount (₹) |
| `loan_term_months` | Repayment period in months |
| `interest_rate` | Applicable interest rate (%) |
| `credit_score` | Credit score (300–900) |
| `existing_loans` | Number of active loans |
| `dti_ratio` | Debt-to-Income ratio (%) |
| `assets_value` | Total assets value (₹) |
| `has_collateral` | Whether collateral is offered (0/1) |
| `co_applicant` | Co-applicant present (0/1) |
| `loan_purpose` | Home / Vehicle / Education / Business / Personal |
| `loan_approved` | **Target** — 1 = Approved, 0 = Rejected |

---

## 🤖 Models Used

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 95.0% | 96.1% | 0.987 |
| Random Forest | 98.0% | 98.5% | 0.996 |
| **Gradient Boosting** | **98.5%** | **98.8%** | **0.995** |

> **Best model: Gradient Boosting Classifier**

---

## 📸 Output Images

13 charts are saved to `outputs/` automatically — EDA charts, confusion matrix, ROC curves, feature importance, model comparison, and cross-validation scores.

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data handling
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualizations
- **scikit-learn** — classifiers, metrics, cross-validation
# loan-approval-predictor
