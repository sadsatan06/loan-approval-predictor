"""
generate_data.py
Generates a realistic loan applicant dataset with approval decisions.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

N = 1000

# ── Demographics ─────────────────────────────────────────────────────────────
age             = np.random.randint(21, 65, N)
gender          = np.random.choice(["Male", "Female"], N, p=[0.58, 0.42])
education       = np.random.choice(
    ["High School", "Bachelor's", "Master's", "PhD"],
    N, p=[0.20, 0.50, 0.22, 0.08]
)
employment      = np.random.choice(
    ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
    N, p=[0.55, 0.25, 0.15, 0.05]
)

# ── Financials ───────────────────────────────────────────────────────────────
annual_income   = np.where(
    employment == "Unemployed",
    np.random.randint(0, 80000, N),
    np.random.randint(200000, 2000000, N)
)
annual_income   = np.clip(annual_income, 0, 5000000)

loan_amount     = np.random.randint(50000, 2000000, N)
loan_term       = np.random.choice([12, 24, 36, 48, 60, 84, 120], N)
interest_rate   = np.round(np.random.uniform(7.5, 18.0, N), 2)

credit_score    = np.clip(
    np.random.normal(680, 90, N).astype(int),
    300, 900
)

existing_loans  = np.random.randint(0, 5, N)
monthly_expenses = np.random.randint(5000, 80000, N)

# EMI estimate
monthly_income   = annual_income / 12
emi              = (loan_amount * (interest_rate/1200) *
                    (1 + interest_rate/1200)**loan_term /
                    ((1 + interest_rate/1200)**loan_term - 1))
emi              = np.round(emi, 0)
dti_ratio        = np.round((emi + monthly_expenses * 0.3) / (monthly_income + 1) * 100, 2)

assets_value     = np.random.randint(0, 5000000, N)
has_collateral   = (assets_value > loan_amount * 1.2).astype(int)
co_applicant     = np.random.choice([0, 1], N, p=[0.55, 0.45])

loan_purpose     = np.random.choice(
    ["Home", "Education", "Vehicle", "Business", "Personal"],
    N, p=[0.30, 0.15, 0.20, 0.20, 0.15]
)

# ── Approval Logic ─────────────────────────────────────────────────────────────
# Score based on key factors
approval_score = (
    (credit_score - 300) / 600 * 40          # 0–40 pts
    + np.clip(1 - dti_ratio / 100, 0, 1) * 25  # 0–25 pts
    + (annual_income > 400000).astype(float) * 15
    + has_collateral * 10
    + co_applicant * 5
    + (employment == "Salaried").astype(float) * 5
    + np.random.normal(0, 5, N)              # noise
)

approved = (approval_score >= 45).astype(int)

# Override: always reject unemployed with low credit
approved[(employment == "Unemployed") & (credit_score < 600)] = 0
# Always reject DTI > 80
approved[dti_ratio > 80] = 0
# Always approve high credit + high income + collateral
approved[(credit_score >= 780) & (annual_income >= 800000) & (has_collateral == 1)] = 1

df = pd.DataFrame({
    "applicant_id":     [f"APP{str(i+1).zfill(4)}" for i in range(N)],
    "age":              age,
    "gender":           gender,
    "education":        education,
    "employment_type":  employment,
    "annual_income":    annual_income,
    "loan_amount":      loan_amount,
    "loan_term_months": loan_term,
    "interest_rate":    interest_rate,
    "credit_score":     credit_score,
    "existing_loans":   existing_loans,
    "monthly_expenses": monthly_expenses,
    "emi_estimate":     emi.astype(int),
    "dti_ratio":        dti_ratio,
    "assets_value":     assets_value,
    "has_collateral":   has_collateral,
    "co_applicant":     co_applicant,
    "loan_purpose":     loan_purpose,
    "loan_approved":    approved,
})

df.to_csv("data/loan_data.csv", index=False)

approved_pct = approved.sum() / N * 100
print(f"✅ Dataset generated: {N} applicants")
print(f"   Approved : {approved.sum()} ({approved_pct:.1f}%)")
print(f"   Rejected : {N - approved.sum()} ({100-approved_pct:.1f}%)")
print(df.head())
