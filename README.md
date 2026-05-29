# FairLoan™ — Enterprise-Grade Kapsayıcı Mikrokredi Skorlama & Risk Yönetim Platformu

FairLoan™, geleneksel bankacılık standartlarının ve katı resmi istihdam bariyerlerinin (teminat veya bordro yetersizliği gibi) finansal sistemin dışına ittiği kadın girişimcileri, mikro üreticileri ve kooperatif üyelerini **alternatif veri analitiği**, **PSD2 Açık Bankacılık entegrasyonları** ve **algoritmik adalet (fairness-audited) ilkeleriyle** bankacılık ekosistemine dahil eden, kurumsal düzeyde bir mikrokredi risk yönetim terminalidir.

Bu kurumsal MVP deposu; önyargılardan arındırılmış makine öğrenmesi (ML) borçlanma modellerini, çift yönlü FastAPI karar API'sini ve sunum/pitch sırasında jüriyi etkilemek üzere tasarlanmış modern, cam tasarımlı (glassmorphic) etkileşimli bir simülatör panelini içerir.

---

## Temel Kurumsal Fintech Modülleri

### 1. Alternatif Net Gelir & DTI Risk Parite Motoru (Modül 1)
* **Kapsayıcı Gelir Modellemesi:** Kayıtlı (formal) geliri bulunmadığı için geleneksel sistemde $0$ TL resmi gelirle değerlendirilip otomatik reddedilen profillerin el emeği üretimleri, e-ticaret ciroları (`eticaret_aylik_ciro`) ve aktif kooperatif emekleri (`kooperatif_uye_ay`) akıllı bir algoritma ile **Alternatif Net Gelir** olarak tescil edilir.
* **DTI (Debt-to-Income) Risk Paritesi:** Kredi geri ödeme taksit tutarı (`aylik_taksit = (limit * 1.30) / 12`) üzerinden borçlanma kapasitesi ölçülür. 
  - *FairLoan:* Alternatif net gelir üzerinden Borç/Gelir (DTI) oranını hesaplar (DTI <= %40 ise "KABUL EDİLEBİLİR").
  - *Geleneksel:* Resmi gelir $0$ TL olduğu için DTI **"HESAPLANAMAZ" (%999+)** durumdadır ve otomatik elenir.

### 2. Kademeli Güven Merdiveni (Progressive Limit Lifecycle - Modül 2)
* **Risk Azaltıcı Yapılandırılmış Kredi Ömrü:** Bankanın ilk aşamadaki temerrüt (default) riskini minimize etmek üzere tasarlanmış, başarılı ödemelerle açılan 4 aşamalı dinamik limit mekanizmasıdır.
  - *Aşama 1 (Başlangıç Mikrokredi):* Toplam limitin %20'si (1-3. Aylar).
  - *Aşama 2 (Girişimci Destek):* Toplam limitin %50'si (4-6. Aylar).
  - *Aşama 3 (Kapasite Artırımı):* Toplam limitin %80'i (7-12. Aylar).
  - *Aşama 4 (Halkbank Partner Portföyü):* Limitinin %100'ü (12+ Ay sonrası - Geleneksel portföye güvenli geçiş aşaması).

### 3. Finansal Sağlık Karnesi & Rapor İhracı (Modül 3)
* **Finansal Sağlık Endeksleri:** Fatura ve kira disiplinini ölçen **Sadakat Endeksi**, ciro üretim gücünü ölçen **Kapasite Endeksi** ve alternatif modelin sağladığı finansal erişim artışını gösteren **Kapsayıcılık Skoru** dinamik halka grafiklerle arayüze yansıtılır.
* **Kurumsal PDF Rapor İhracı:** Tek tıkla tarayıcının yazdırma motorunu tetikleyen, `@media print` mizanpaj kuralları sayesinde gereksiz tüm UI elementlerini otomatik gizleyerek beyaz kağıda basıma hazır, Halkbank logolu, cryptographic hash imzalı ve ıslak imza alanlı resmi bir Kredi Karar Raporu üretir.

### 4. BKM TR-API Sandbox & Açık Bankacılık Consent Hub
* **Regülasyon Uyumlu Sandbox:** Merkez Bankası ve BKM Açık Bankacılık standartlarına uyumlu PSD2 SSL bağlantı protokollerini, NVİ KPS (Mernis) TCKN doğrulamalarını ve veri bütünlüğü onaylarını simüle eden güvenli bir Açık Bankacılık Consent Gateway altyapısı içerir.
* **Kriptografik Bütünlük:** Konsolda ve raporda sergilenen SHA-256 doğrulama imzaları (`sha256_integrity`) ile kararların geriye dönük manipüle edilemezliği kanıtlanır.

---

## Proje Klasör Yapısı

```text
fairloan/
├── data/
│   └── loan_data.csv            # TÜİK labor ve BDDK dağılımlı 2,000 kişilik sentetik veri seti
├── model/
│   ├── fairloan_model.py        # ML eğitim pipeline'ı, bias audit denetimi ve JSON çıktısı
│   └── fairloan_pipeline.pkl    # Eğitilmiş ve serileştirilmiş Gradient Boosting ML modeli
├── api/
│   └── main.py                  # FastAPI skorlama motoru, TR-API Sandbox ve DTI parite endpoints
├── demo/
│   ├── index.html               # Premium, responsive, glassmorphic tek sayfalık simülatör arayüzü
│   └── bias_comparison.png      # ML Bias Audit grafiği (Arayüzde dinamik sergilenen adalet raporu)
├── requirements.txt             # Gerekli kütüphaneler listesi (Kurumsal bağımlılıklar)
└── README.md                    # Bu doküman & Sunum rehberi
```

---

## Hızlı Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
Finansal veri analizi ve web servis bağımlılıklarını kurun:
```bash
pip install -r requirements.txt
```

### 2. Sentetik Veri Setini Üretme (TÜİK Demografik Uyumlu)
TÜİK hanehalkı kadın işgücü istatistiklerine göre ağırlıklandırılmış sentetik veri setini oluşturun:
```bash
python data_generator.py
```

### 3. ML Model Eğitimi & Algoritmik Bias Auditing
Modeli eğitin, ROC-AUC doğruluğunu görün ve disparate impact adalet katsayılarını otomatik ihraç edin:
```bash
python model/fairloan_model.py
```
*Eğitim bittiğinde `model/fairness_metrics.json` ve bias kıyaslama grafiği otomatik üretilir. Geleneksel sistemin önyargılı onay oranı %0.0 iken FairLoan'un disparate impact oranı **1.01** (mükemmel adil dağılım) olarak hesaplanır.*

### 4. FastAPI Sunucusunu Başlatma
Karar destek API'sini port 8000 üzerinde ayağa kaldırın:
```bash
uvicorn api.main:app
```

### 5. Simülatörü Başlatma
Sunucu aktifken tarayıcınızda [http://127.0.0.1:8000](http://127.0.0.1:8000) adresini açarak platformu kullanmaya başlayabilirsiniz.
*(Not: Sunucu kapalı olsa bile arayüze entegre yerel fallback motoru sayesinde simülasyon tüm finansal sağlık endekslerini ve kararlarını sıfır gecikmeyle hesaplamaya devam eder!)*

---

## Teknolojiler & Kütüphaneler

* **Veri & Analitik:** Pandas, NumPy, Matplotlib
* **Yapay Zeka (ML Hattı):** Scikit-Learn (Gradient Boosting Classifier, StandardScaler, Pipeline), Joblib
* **Kurumsal Backend:** FastAPI, Pydantic, Uvicorn
* **Gelişmiş Ön Yüz:** HTML5, Vanilla CSS3 (Glassmorphism, Neon Glow FX, @media print mizanpaj yönetimi), Modern Javascript ES6+ (Count-up animasyonları, SVG progress rings)
