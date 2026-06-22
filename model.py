"""
model.py
Trains multiple classifiers to predict loan approval.
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
Outputs: confusion matrix, ROC curve, feature importance, model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/loan_data.csv")

# ── Encode categoricals ───────────────────────────────────────────────────────
cat_cols = ["gender", "education", "employment_type", "loan_purpose"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

FEATURES = [
    "age", "gender_enc", "education_enc", "employment_type_enc",
    "annual_income", "loan_amount", "loan_term_months", "interest_rate",
    "credit_score", "existing_loans", "monthly_expenses", "emi_estimate",
    "dti_ratio", "assets_value", "has_collateral", "co_applicant",
    "loan_purpose_enc"
]
FEAT_LABELS = [
    "Age", "Gender", "Education", "Employment",
    "Annual Income", "Loan Amount", "Loan Term", "Interest Rate",
    "Credit Score", "Existing Loans", "Monthly Expenses", "EMI Estimate",
    "DTI Ratio", "Assets Value", "Collateral", "Co-Applicant",
    "Loan Purpose"
]

X = df[FEATURES]
y = df["loan_approved"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train Models ─────────────────────────────────────────────────────────────
models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
}

results = {}
print("=" * 60)
print("  MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred      = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, pred_prob)
    roc_auc = auc(fpr, tpr)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    results[name] = {
        "model": model, "pred": pred, "pred_prob": pred_prob,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "fpr": fpr, "tpr": tpr, "auc": roc_auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    }
    print(f"\n  {name}")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    CV Acc    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

best_name  = max(results, key=lambda n: results[n]["f1"])
best       = results[best_name]
print(f"\n  ⭐ Best model: {best_name}  (F1 = {best['f1']*100:.2f}%)")
print(f"\n  Classification Report ({best_name}):")
print(classification_report(y_test, best["pred"], target_names=["Rejected", "Approved"]))

# ── 09: Confusion Matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, best["pred"])
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"])
disp.plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/09_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✅ Saved: 09_confusion_matrix.png")

# ── 10: ROC Curves (all models) ───────────────────────────────────────────────
COLORS = {"Logistic Regression": "#3498db", "Random Forest": "#2ecc71", "Gradient Boosting": "#e74c3c"}
fig, ax = plt.subplots(figsize=(8, 7))
for name, res in results.items():
    ax.plot(res["fpr"], res["tpr"],
            color=COLORS[name], linewidth=2.2,
            label=f"{name}  (AUC = {res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Random Classifier")
ax.fill_between(results[best_name]["fpr"], results[best_name]["tpr"], alpha=0.08, color=COLORS[best_name])
ax.set_title("ROC Curves — All Models", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.legend(fontsize=11, loc="lower right")
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/10_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 10_roc_curves.png")

# ── 11: Feature Importance (best tree model) ──────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_idx)))

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh([FEAT_LABELS[i] for i in sorted_idx], importances[sorted_idx],
        color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("Feature Importance for Loan Approval Prediction", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="x", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/11_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 11_feature_importance.png")

# ── 12: Model Comparison Dashboard ───────────────────────────────────────────
metrics_map = {
    "Accuracy":  "acc",
    "Precision": "prec",
    "Recall":    "rec",
    "F1 Score":  "f1",
}
short = ["LR", "RF", "GB"]
model_names_list = list(results.keys())
bar_colors = ["#3498db", "#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for i, (metric_label, metric_key) in enumerate(metrics_map.items()):
    ax = axes[i]
    vals = [results[n][metric_key] * 100 for n in model_names_list]
    bars = ax.bar(short, vals, color=bar_colors, edgecolor="white", linewidth=1, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(metric_label, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Model Performance Comparison  (LR = Logistic, RF = Random Forest, GB = Gradient Boosting)",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/12_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 12_model_comparison.png")

# ── 13: Cross-Validation Scores ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(model_names_list))
cv_means = [results[n]["cv_mean"] * 100 for n in model_names_list]
cv_stds  = [results[n]["cv_std"]  * 100 for n in model_names_list]
bars = ax.bar(x, cv_means, color=bar_colors, edgecolor="white", linewidth=1,
              width=0.5, yerr=cv_stds, capsize=6, error_kw={"linewidth": 2, "color": "black"})
for bar, v in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Logistic\nRegression", "Random\nForest", "Gradient\nBoosting"], fontsize=11)
ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=15, fontweight="bold", pad=14)
ax.set_ylabel("CV Accuracy (%)", fontsize=12)
ax.set_ylim(0, 112)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/13_cross_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Saved: 13_cross_validation.png")

print("\n✅ All model outputs saved to outputs/")
print(f"\n{'─'*60}")
print(f"  SUMMARY")
print(f"{'─'*60}")
for name in model_names_list:
    r = results[name]
    print(f"  {name:<25}  Acc: {r['acc']*100:.1f}%  F1: {r['f1']*100:.1f}%  AUC: {r['auc']:.3f}")
print(f"{'─'*60}")
print(f"  ⭐ Best → {best_name}")
print(f"{'─'*60}")
