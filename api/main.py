from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="FairLoan API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained ML pipeline
model_path = "model/fairloan_pipeline.pkl"
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Successfully loaded trained model from '{model_path}'")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
else:
    print(f"Warning: Model not found at '{model_path}'. Please run the ML pipeline training script first.")

class ApplicantData(BaseModel):
    gelir: float
    istihdam_turu: str
    fatura_odeme_skoru: float
    kira_gecmis_ay: int
    eticaret_aylik_ciro: float
    kooperatif_uye_ay: int
    sosyal_ag_skoru: float

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Attempt to serve demo/index.html
    paths_to_try = ["demo/index.html", "../demo/index.html", "index.html"]
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return "<h1>FairLoan Demo Page Not Found</h1><p>Please check the project directory structure.</p>"

@app.get("/bias_comparison.png")
def get_bias_comparison():
    paths_to_try = ["demo/bias_comparison.png", "bias_comparison.png"]
    for path in paths_to_try:
        if os.path.exists(path):
            return FileResponse(path)
    return "Image not found"

@app.post("/score")
def score_applicant(data: ApplicantData):
    # 1. Geleneksel Bankacılık Kararı (Biased Rule)
    # Geleneksel sistemde gelir > 10,000 ve formal (sigortalı/kadrolu) çalışan olmak zorunludur.
    geleneksel_onay = bool((data.gelir > 10000) and (data.istihdam_turu == "formal"))
    
    geleneksel_sebepler = []
    if data.gelir <= 10000:
        geleneksel_sebepler.append(f"Yetersiz Aylık Gelir ({data.gelir:,.0f} TL < 10,000 TL barajı)")
    if data.istihdam_turu != "formal":
        istihdam_tr = {
            "serbest": "Güvencesiz / Serbest Çalışan",
            "ev_kadinı": "Kayıt Dışı / Ev Hanımı",
            "kooperatif": "Alternatif İstihdam / Kooperatif Üyesi"
        }.get(data.istihdam_turu, data.istihdam_turu)
        geleneksel_sebepler.append(f"Kabul Edilmeyen İstihdam Türü ({istihdam_tr})")
    
    if geleneksel_onay:
        geleneksel_sebepler.append("Resmi istihdam türü ve yeterli gelir seviyesi onaylandı.")

    # 2. FairLoan ML Kararı (Alternative & Inclusive Scoring)
    # Calculate a smooth raw score based on feature contributions (0 to 1000 points)
    # 1. Fatura Ödeme (Max 300 points) -> score * 3.0
    fatura_pts = data.fatura_odeme_skoru * 3.0
    
    # 2. Kira Geçmişi (Max 200 points) -> 200 points for 48+ months
    kira_pts = min(data.kira_gecmis_ay / 48.0, 1.0) * 200.0
    
    # 3. E-Ticaret Ciro (Max 300 points) -> 300 points for 10,000+ TL monthly ciro
    ciro_pts = min(data.eticaret_aylik_ciro / 10000.0, 1.0) * 300.0
    
    # 4. Kooperatif Üyelik (Max 150 points) -> 150 points for 24+ months
    koop_pts = min(data.kooperatif_uye_ay / 24.0, 1.0) * 150.0
    
    # 5. Sosyal Ağ Skoru (Max 50 points) -> score * 0.5
    sosyal_pts = data.sosyal_ag_skoru * 0.5
    
    raw_score = fatura_pts + kira_pts + ciro_pts + koop_pts + sosyal_pts # Total Max: 1000
    
    prob = 0.0
    onay = False
    score = 0
    factors = []
    
    if model is not None:
        try:
            # Model features: fatura_odeme_skoru, kira_gecmis_ay, eticaret_aylik_ciro, kooperatif_uye_ay, sosyal_ag_skoru
            features = [[
                data.fatura_odeme_skoru,
                data.kira_gecmis_ay,
                data.eticaret_aylik_ciro,
                data.kooperatif_uye_ay,
                data.sosyal_ag_skoru
            ]]
            prob = float(model.predict_proba(features)[0][1])
            onay = bool(prob >= 0.5)
        except Exception as e:
            print(f"Error during prediction: {e}")
            prob = 0.5
            onay = True
    else:
        # Heuristic decision
        has_good_bill = data.fatura_odeme_skoru > 70
        has_good_rent = data.kira_gecmis_ay > 12
        has_good_ecommerce = data.eticaret_aylik_ciro > 2000
        has_good_coop = data.kooperatif_uye_ay > 6
        onay = bool(has_good_bill or has_good_rent or has_good_ecommerce or has_good_coop)
        prob = 0.85 if onay else 0.25

    # Calibrate final score smoothly using 300 - 980 standard credit rating scale
    # If ML prediction is positive, score is in 620-980 range. If negative, in 300-590 range.
    if onay:
        score = round(620 + (raw_score / 1000.0) * 360)
    else:
        score = round(300 + (raw_score / 1000.0) * 290)
        
    score = min(max(score, 300), 980) # Clamp score for realistic credit ratings

    # 3. Dinamik Açıklama Faktörleri
    factors = _explain(data, prob)
    risk_seviyesi = "Düşük Risk" if score > 700 else "Orta Risk" if score > 400 else "Yüksek Risk"

    # 4. AI Score Builder (Skor İyileştirme Planı)
    # Calibrate realistic targets based on current score
    target_score = min(score + 180, 850) if score < 700 else min(score + 80, 950)
    if target_score < 620:
        target_score = 680
        
    limit = "15.000" if target_score < 700 else "25.000" if target_score < 800 else "50.000"
    
    # Identify key improvement areas
    actions = []
    if data.fatura_odeme_skoru < 80:
        actions.append("fatura ödemelerinizi düzenli yapıp")
    if data.kira_gecmis_ay < 12:
        actions.append("banka üzerinden düzenli kira ödemelerinizi belgeleyip")
    if data.eticaret_aylik_ciro < 2500:
        actions.append("e-ticaret aylık cironuzu artırıp")
    if data.kooperatif_uye_ay < 6:
        actions.append("kadın kooperatifi teslimatlarınızı sisteme girip")
    if data.sosyal_ag_skoru < 75:
        actions.append("sosyal referans puanlarınızı güçlendirip")
        
    if not actions:
        actions = ["fatura disiplininizi koruyup", "aktif ekonomik katılımınızı sürdürüp"]
        
    if len(actions) > 2:
        action_str = ", ".join(actions[:-1]) + " ve " + actions[-1]
    elif len(actions) == 2:
        action_str = f"{actions[0]} ve {actions[1]}"
    else:
        action_str = actions[0]
        
    score_builder_text = f"Skorunuz şu an {score}. Sonraki 3 ay boyunca {action_str} durumunda skorunuz {target_score}'e yükselecek ve {limit} TL limitli mikro-krediniz onaylanacaktır."

    # 5. TÜİK & BDDK Real Data Calibration (Hybrid Approach)
    # Average monthly wage in TR is calibrated at ~18,000 TL for baseline female workforce
    gelir_vs_avg = round(((data.gelir / 18000) - 1) * 100)
    
    # Female employment stat based on choice (TÜİK & TBB)
    istihdam_stat = ""
    if data.istihdam_turu == "ev_kadinı":
        istihdam_stat = "TÜİK Hanehalkı İşgücü Araştırması'na göre Türkiye'deki kadınların %64.9'u resmi olarak işgücüne katılamamaktadır. Ev içi el emeği üretimi finansal olarak tamamen görünmezdir."
    elif data.istihdam_turu == "serbest":
        istihdam_stat = "TÜİK verilerine göre serbest veya freelance çalışan kadınların %41.2'si güvencesiz ve kayıt dışı istihdam grubundadır, bu sebeple geleneksel sistemde kredi dışı bırakılırlar."
    elif data.istihdam_turu == "kooperatif":
        istihdam_stat = "TBB verilerine göre kooperatif üyesi kadın üreticilerin geleneksel bankacılık kredi sistemine erişim oranı sadece %8.4 seviyesindedir."
    else:
        istihdam_stat = "TÜİK verilerine göre formal çalışan kadınların ortalama maaşı, erkek meslektaşlarına kıyasla %15.6 oranında 'cinsiyet ücret uçurumu' (Gender Pay Gap) adaletsizliğine maruz kalmaktadır."

    # Bill payment vs TBB Risk Center average (74/100)
    fatura_vs_avg = "ortalamanın üzerinde" if data.fatura_odeme_skoru >= 74 else "ortalamanın altında"
    fatura_diff = round(abs(data.fatura_odeme_skoru - 74))
    
    # E-commerce revenue vs KOSGEB average (3,500 TL)
    ciro_vs_avg = round(((data.eticaret_aylik_ciro / 3500) - 1) * 100) if data.eticaret_aylik_ciro > 0 else -100

    # BDDK Gender credit distribution
    bddk_stat = "BDDK istatistiklerine göre Türkiye'de bireysel kredilerin sadece %28.4'ü kadınlara tahsis edilmektedir. Geleneksel skorlama kadın aleyhinedir."

    calibration_report = {
        "gelir_vs_avg": gelir_vs_avg,
        "istihdam_stat": istihdam_stat,
        "fatura_vs_avg": fatura_vs_avg,
        "fatura_diff": fatura_diff,
        "ciro_vs_avg": ciro_vs_avg,
        "bddk_stat": bddk_stat
    }

    return {
        "fairloan": {
            "kredi_skoru": score,
            "onay": onay,
            "risk_seviyesi": risk_seviyesi,
            "aciklama": factors,
            "skor_iyilestirme_plani": score_builder_text,
            "kalibrasyon": calibration_report
        },
        "geleneksel": {
            "onay": geleneksel_onay,
            "sebepler": geleneksel_sebepler
        }
    }

def _explain(data, prob):
    factors = []
    
    # Positive drivers
    if data.fatura_odeme_skoru >= 75:
        factors.append(f"Düzenli fatura ödeme geçmişi (Fatura Skoru: {data.fatura_odeme_skoru:.0f}/100)")
    if data.kira_gecmis_ay >= 12:
        factors.append(f"Düzenli kira ödeme geçmişi ({data.kira_gecmis_ay} ay kesintisiz)")
    if data.eticaret_aylik_ciro >= 2000:
        factors.append(f"Aktif e-ticaret mikro-geliri ({data.eticaret_aylik_ciro:,.0f} TL/ay)")
    if data.kooperatif_uye_ay >= 6:
        factors.append(f"Aktif kadın kooperatifi üyeliği ({data.kooperatif_uye_ay} ay)")
    if data.sosyal_ag_skoru >= 75:
        factors.append(f"Yüksek yerel dayanışma/sosyal sermaye skoru ({data.sosyal_ag_skoru:.0f}/100)")
        
    # Negative drivers / risks if low score
    if prob < 0.5:
        if data.fatura_odeme_skoru < 70:
            factors.append(f"Geliştirilmesi gereken fatura ödeme disiplini ({data.fatura_odeme_skoru:.0f}/100)")
        if data.kira_gecmis_ay < 6:
            factors.append(f"Çok kısa veya belgelenemeyen kira geçmişi ({data.kira_gecmis_ay} ay)")
        if data.eticaret_aylik_ciro < 1000 and data.kooperatif_uye_ay < 3:
            factors.append("Yetersiz mikro-girişimsel ve kooperatif gelir kanıtı")

    return factors
