from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)
os.makedirs("demo", exist_ok=True)

df = pd.read_csv("data/loan_data.csv")

ALTERNATIVE_FEATURES = [
    "fatura_odeme_skoru", "kira_gecmis_ay",
    "eticaret_aylik_ciro", "kooperatif_uye_ay", "sosyal_ag_skoru"
]
TARGET = "fairloan_onay"

X = df[ALTERNATIVE_FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42))
])
pipeline.fit(X_train, y_train)

# Predict on whole dataset for audit
df["tahmin"] = pipeline.predict(X)

# Fairness Audit Calculations
def audit_fairness(dataframe, prediction_col, label_col):
    # Privileged Group (Formal)
    privileged_mask = (dataframe["istihdam_turu"] == "formal")
    privileged_total = privileged_mask.sum()
    privileged_approved = dataframe[privileged_mask][prediction_col].sum()
    privileged_rate = privileged_approved / privileged_total if privileged_total > 0 else 0.0

    # Unprivileged Groups
    unprivileged_groups = ["ev_kadinı", "serbest", "kooperatif"]
    results = {"rates": {"formal": round(privileged_rate, 4)}, "disparate_impact": {}}
    
    # Calculate Disparate Impact Ratio
    for group in unprivileged_groups:
        mask = (dataframe["istihdam_turu"] == group)
        total = mask.sum()
        approved = dataframe[mask][prediction_col].sum()
        group_rate = approved / total if total > 0 else 0.0
        results["rates"][group] = round(group_rate, 4)
        
        # Disparate Impact Ratio = Group Rate / Privileged Rate
        dir_val = group_rate / privileged_rate if privileged_rate > 0 else 0.0
        results["disparate_impact"][group] = round(min(dir_val, 5.0), 4)

    return results

# Audit both Traditional rules and FairLoan ML model
trad_audit = audit_fairness(df, "geleneksel_onay", "geleneksel_onay")
fair_audit = audit_fairness(df, "tahmin", "fairloan_onay")

# Print results
roc_auc = float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1]))
print(f"ROC-AUC: {roc_auc:.4f}")
print("Traditional Fairness Audit:", json.dumps(trad_audit, indent=2))
print("FairLoan Fairness Audit:", json.dumps(fair_audit, indent=2))

# Export fairness metrics to JSON for API delivery
fairness_metrics_export = {
    "roc_auc": round(roc_auc * 100, 2),
    "traditional": {
        "ev_kadinı_onay": round(trad_audit["rates"]["ev_kadinı"] * 100, 2),
        "serbest_onay": round(trad_audit["rates"]["serbest"] * 100, 2),
        "formal_onay": round(trad_audit["rates"]["formal"] * 100, 2),
        "disparate_impact_ev_kadinı": round(trad_audit["disparate_impact"]["ev_kadinı"], 2),
        "disparate_impact_serbest": round(trad_audit["disparate_impact"]["serbest"], 2)
    },
    "fairloan": {
        "ev_kadinı_onay": round(fair_audit["rates"]["ev_kadinı"] * 100, 2),
        "serbest_onay": round(fair_audit["rates"]["serbest"] * 100, 2),
        "formal_onay": round(fair_audit["rates"]["formal"] * 100, 2),
        "disparate_impact_ev_kadinı": round(fair_audit["disparate_impact"]["ev_kadinı"], 2),
        "disparate_impact_serbest": round(fair_audit["disparate_impact"]["serbest"], 2)
    }
}

with open("model/fairness_metrics.json", "w", encoding="utf-8") as f:
    json.dump(fairness_metrics_export, f, indent=2)
print("Fairness metrics saved successfully to 'model/fairness_metrics.json'.")

# Save the trained model pipeline
joblib.dump(pipeline, "model/fairloan_pipeline.pkl")
print("Model pipeline saved successfully to 'model/fairloan_pipeline.pkl'.")

# Generate Bias Comparison Plot
# Traditional vs FairLoan approval rates across employment types
plt.figure(figsize=(10, 6))
compare = df.groupby("istihdam_turu")[["geleneksel_onay", "fairloan_onay"]].mean()

# Rename keys for a prettier visual representation
compare = compare.rename(index={
    "formal": "Formal Çalışan",
    "serbest": "Serbest / Freelance",
    "ev_kadinı": "Ev Hanımı",
    "kooperatif": "Kooperatif Üyesi"
})

ax = compare.plot(kind="bar", color=["#64748B", "#10B981"], figsize=(10, 6))
plt.title("Onay Oranı Karşılaştırması: Geleneksel vs. FairLoan", fontsize=14, fontweight="bold", pad=15)
plt.ylabel("Onay Oranı", fontsize=12)
plt.xlabel("İstihdam Türü", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(["Geleneksel Model (Dışlayıcı)", "FairLoan Model (Kapsayıcı)"], fontsize=10)

# Add values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("bias_comparison.png", dpi=300)
plt.savefig("demo/bias_comparison.png", dpi=300)
print("Bias comparison chart generated and saved to 'bias_comparison.png' and 'demo/bias_comparison.png'.")
