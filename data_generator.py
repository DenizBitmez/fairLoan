import pandas as pd
import numpy as np
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    # Geleneksel (cinsiyet yanlılığı içeren)
    "gelir":            np.where(np.random.rand(n) < 0.5,
                            np.random.normal(12000, 3000, n),   # erkek
                            np.random.normal(9000, 3000, n)),   # kadın
    "istihdam_turu":    np.random.choice(
                            ["formal","serbest","ev_kadinı","kooperatif"], n,
                            p=[0.4, 0.25, 0.2, 0.15]),

    # Alternatif (FairLoan'ın baktığı)
    "fatura_odeme_skoru":   np.random.randint(60, 100, n),      # 0-100
    "kira_gecmis_ay":       np.random.randint(0, 60, n),
    "eticaret_aylik_ciro":  np.random.exponential(3000, n),
    "kooperatif_uye_ay":    np.random.randint(0, 120, n),
    "sosyal_ag_skoru":      np.random.randint(0, 100, n),

    # Hedef — geleneksel sistemde kadınlar daha az onay alıyor
    # bunu kasıtlı olarak simüle ediyoruz
})

# Geleneksel label: gelir + istihdam_turu ağırlıklı → kadın aleyhine
df["geleneksel_onay"] = (
    (df["gelir"] > 10000) &
    (df["istihdam_turu"] == "formal")
).astype(int)

# FairLoan label: alternatif verilerle daha kapsayıcı
df["fairloan_onay"] = (
    (df["fatura_odeme_skoru"] > 70) |
    (df["kira_gecmis_ay"] > 12) |
    (df["eticaret_aylik_ciro"] > 2000) |
    (df["kooperatif_uye_ay"] > 6)
).astype(int)

df.to_csv("data/loan_data.csv", index=False)
print("Synthetic data generated successfully at 'data/loan_data.csv'.")
