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
bias_report = df.groupby("istihdam_turu")["tahmin"].mean().to_dict()

# Print results
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc_auc:.4f}")
print("Bias Report:", json.dumps(bias_report, indent=2))

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
