import pandas as pd
import numpy as np
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    # 1. TÜİK İşgücü göstergelerine göre istihdam türü olasılıkları (p değerleri)
    # Türkiye'deki kadınların işgücüne katılım oranına ve kayıt dışı/kooperatif üretimine uygun dağılım:
    # Kadınların büyük kısmı resmi işgücünde yer alamayıp ev hanımı/kooperatif/serbest gruptadır.
    "istihdam_turu":    np.random.choice(
                            ["formal", "serbest", "ev_kadinı", "kooperatif"], n,
                            p=[0.35, 0.20, 0.38, 0.07]),
})

# 2. İstihdam türüne ve TÜİK/ILO cinsiyet gelir eşitsizliği verilerine göre gelir dağılımı
# Ev hanımlarının resmi geliri 0 TL'dir (bu durum onları geleneksel bankacılıkta doğrudan eler).
# Formal çalışanlar Türkiye ortalama kadın maaşı düzeyindedir (varyanslı).
gelirler = []
for i in range(n):
    emp = df.loc[i, "istihdam_turu"]
    if emp == "ev_kadinı":
        gelir = 0.0
    elif emp == "formal":
        gelir = np.random.normal(24000, 4500)
    elif emp == "serbest":
        gelir = np.random.normal(15000, 3500)
    else:  # kooperatif
        gelir = np.random.normal(9500, 2000)
    gelirler.append(max(0.0, gelir))

df["gelir"] = gelirler

# Alternatif Veriler (FairLoan'ın baktığı alternatif finansal güvenilirlik)
# Ev hanımları ve serbest çalışanlar resmi geliri az olsa dahi fatura disiplinine, kira geçmişine,
# e-ticaret satış cirolarına veya kooperatif emeğine sahip olabilir.
df["fatura_odeme_skoru"] = np.random.randint(55, 100, n)
df["kira_gecmis_ay"] = np.random.randint(0, 60, n)
df["eticaret_aylik_ciro"] = np.random.exponential(2800, n)
df["kooperatif_uye_ay"] = np.random.randint(0, 120, n)
df["sosyal_ag_skoru"] = np.random.randint(30, 100, n)

# Geleneksel label: Yüksek resmi gelir (>18,000 TL) ve resmi istihdam (formal) zorunluluğu
df["geleneksel_onay"] = (
    (df["gelir"] > 18000) &
    (df["istihdam_turu"] == "formal")
).astype(int)

# FairLoan label: Alternatif veriler aracılığıyla finansal sorumluluk ve kapsayıcı kredi onayı
# Fatura disiplini (>75) veya düzenli kira geçmişi (>12 ay) veya aktif mikro e-ticaret cirosu (>2200 TL)
# ya da kadın kooperatifinde sürekli katılım (>6 ay) durumunda onaylanır.
df["fairloan_onay"] = (
    ((df["fatura_odeme_skoru"] > 74) & (df["kira_gecmis_ay"] > 12)) |
    (df["eticaret_aylik_ciro"] > 2200) |
    (df["kooperatif_uye_ay"] > 6)
).astype(int)

df.to_csv("data/loan_data.csv", index=False)
print("Synthetic data generated successfully at 'data/loan_data.csv'.")
