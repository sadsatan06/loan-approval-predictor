"""
visualize.py
Generates all exploratory analysis charts for loan approval data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/loan_data.csv")

GREEN  = "#2ecc71"
RED    = "#e74c3c"
BLUE   = "#3498db"
PURPLE = "#9b59b6"
ORANGE = "#f39c12"

def status_colors(series):
    return [GREEN if v == 1 else RED for v in series]

legend_patches = [
    mpatches.Patch(color=GREEN, label="Approved"),
    mpatches.Patch(color=RED,   label="Rejected"),
]

print("📊 Generating visualizations...")

# ── 01: Approval Rate Pie ────────────────────────────────────────────────────
counts = df["loan_approved"].value_counts()
fig, ax = plt.subplots(figsize=(7, 6))
ax.pie([counts[1], counts[0]],
       labels=["Approved", "Rejected"],
       autopct="%1.1f%%",
       colors=[GREEN, RED],
       startangle=140,
       wedgeprops=dict(edgecolor="white", linewidth=2.5),
       pctdistance=0.80,
       textprops={"fontsize": 13})
ax.set_title("Overall Loan Approval Rate", fontsize=16, fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("outputs/01_approval_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 01_approval_rate.png")

# ── 02: Approval Rate by Employment Type ─────────────────────────────────────
emp_approval = df.groupby("employment_type")["loan_approved"].mean() * 100
fig, ax = plt.subplots(figsize=(9, 5))
colors = [GREEN if v >= 50 else RED for v in emp_approval.values]
bars = ax.bar(emp_approval.index, emp_approval.values, color=colors,
              edgecolor="white", linewidth=1.2, width=0.55)
for bar, val in zip(bars, emp_approval.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="50% line")
ax.set_title("Loan Approval Rate by Employment Type", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Employment Type", fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_ylim(0, 105)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/02_approval_by_employment.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 02_approval_by_employment.png")

# ── 03: Credit Score Distribution by Approval ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for status, color, label in [(1, GREEN, "Approved"), (0, RED, "Rejected")]:
    subset = df[df["loan_approved"] == status]["credit_score"]
    ax.hist(subset, bins=30, color=color, alpha=0.65, label=label, edgecolor="white", linewidth=0.5)
ax.axvline(df["credit_score"].mean(), color="navy", linewidth=1.8,
           linestyle="--", label=f"Mean ({df['credit_score'].mean():.0f})")
ax.set_title("Credit Score Distribution: Approved vs Rejected", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Credit Score", fontsize=12)
ax.set_ylabel("Number of Applicants", fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/03_credit_score_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 03_credit_score_dist.png")

# ── 04: Annual Income vs Loan Amount (scatter) ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for status, color, label in [(1, GREEN, "Approved"), (0, RED, "Rejected")]:
    sub = df[df["loan_approved"] == status]
    ax.scatter(sub["annual_income"] / 1e5, sub["loan_amount"] / 1e5,
               color=color, alpha=0.45, s=30, label=label, edgecolors="none")
ax.set_title("Annual Income vs Loan Amount", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Annual Income (₹ Lakhs)", fontsize=12)
ax.set_ylabel("Loan Amount (₹ Lakhs)", fontsize=12)
ax.legend(fontsize=11)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/04_income_vs_loan.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 04_income_vs_loan.png")

# ── 05: DTI Ratio Boxplot by Approval ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
data_bp = [df[df["loan_approved"] == 1]["dti_ratio"].values,
           df[df["loan_approved"] == 0]["dti_ratio"].values]
bp = ax.boxplot(data_bp, patch_artist=True, tick_labels=["Approved", "Rejected"],
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], [GREEN, RED]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_title("Debt-to-Income Ratio: Approved vs Rejected", fontsize=15, fontweight="bold", pad=14)
ax.set_ylabel("DTI Ratio (%)", fontsize=12)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/05_dti_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 05_dti_boxplot.png")

# ── 06: Approval by Loan Purpose ─────────────────────────────────────────────
purpose_rate = df.groupby("loan_purpose")["loan_approved"].mean() * 100
purpose_rate = purpose_rate.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 5))
colors_p = [GREEN if v >= 60 else ORANGE if v >= 50 else RED for v in purpose_rate.values]
bars = ax.barh(purpose_rate.index, purpose_rate.values, color=colors_p,
               edgecolor="white", linewidth=1, height=0.5)
for bar, val in zip(bars, purpose_rate.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
ax.set_title("Approval Rate by Loan Purpose", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Approval Rate (%)", fontsize=12)
ax.set_xlim(0, 110)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="x", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/06_approval_by_purpose.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 06_approval_by_purpose.png")

# ── 07: Approval by Education ────────────────────────────────────────────────
edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
edu_rate  = df.groupby("education")["loan_approved"].mean() * 100
edu_rate  = edu_rate.reindex([e for e in edu_order if e in edu_rate.index])
fig, ax = plt.subplots(figsize=(9, 5))
colors_e = [GREEN if v >= 60 else ORANGE if v >= 50 else RED for v in edu_rate.values]
bars = ax.bar(edu_rate.index, edu_rate.values, color=colors_e,
              edgecolor="white", linewidth=1.2, width=0.5)
for bar, val in zip(bars, edu_rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Approval Rate by Education Level", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Education", fontsize=12)
ax.set_ylabel("Approval Rate (%)", fontsize=12)
ax.set_ylim(0, 105)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/07_approval_by_education.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 07_approval_by_education.png")

# ── 08: Correlation Heatmap ───────────────────────────────────────────────────
num_cols = ["age", "annual_income", "loan_amount", "credit_score",
            "dti_ratio", "existing_loans", "has_collateral",
            "co_applicant", "assets_value", "loan_approved"]
short_labels = ["Age", "Income", "Loan Amt", "Credit Score",
                "DTI", "Existing Loans", "Collateral",
                "Co-Applicant", "Assets", "Approved"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.5, square=True, ax=ax, annot_kws={"size": 9},
            vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=14)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 08_correlation_heatmap.png")

print("\n✅ All 8 visualizations saved to outputs/")
