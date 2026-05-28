# FairLoan MVP — Kapsayıcı Mikrokredi Skorlama Platformu

FairLoan, geleneksel bankacılık sistemlerinin teminat veya kayıtlı (formal) istihdam yetersizliği sebebiyle finans dünyasının dışına ittiği kadın girişimci, kooperatif üyesi ve mikro üreticileri **alternatif veri kaynakları** ve **önyargısız makine öğrenmesi** modelleri ile finansal sisteme kazandıran kapsayıcı bir mikrokredi skorlama platformudur.

Bu depo, FairLoan projesinin ML hattını, FastAPI tabanlı çift yönlü skorlama API'sini ve pitch deck sunumları sırasında canlı olarak kullanabileceğiniz modern ve etkileşimli bir simülatör arayüzünü içerir.

---

## Proje Klasör Yapısı

```text
fairloan/
├── data/
│   └── loan_data.csv            # Üretilen 2,000 kişilik sentetik veri seti
├── model/
│   ├── fairloan_model.py        # ML eğitim pipeline'ı ve Bias Audit kodları
│   └── fairloan_pipeline.pkl    # Eğitilmiş ve serileştirilmiş Gradient Boosting modeli
├── api/
│   └── main.py                  # FastAPI skorlama endpoint'i ve karşılaştırma mantığı
├── demo/
│   ├── index.html               # Tek dosyada toplanmış modern, glassmorphic UI simülatörü
│   └── bias_comparison.png      # Otomatik üretilen önyargı denetim grafiği (UI entegreli)
├── bias_comparison.png          # Kök dizindeki grafik kopyası
├── requirements.txt             # Gerekli kütüphaneler listesi
└── README.md                    # Bu doküman & Sunum senaryosu
```

---

## Hızlı Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
Öncelikle gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Sentetik Veriyi Üretme
Geleneksel bankacılık sistemlerindeki önyargıları ve FairLoan'ın alternatif veri yapısını yansıtan sentetik veri setini oluşturun:
```bash
python data_generator.py
```

### 3. ML Modelini Eğitme ve Adalet Denetimi (Bias Audit)
Gradient Boosting sınıflandırıcısını eğitmek, model metriğini görmek ve önyargı analizi grafiğini otomatik oluşturmak için çalıştırın:
```bash
python model/fairloan_model.py
```
*Bu komut sonucunda ekrana **ROC-AUC skoru** ile istihdam türlerine göre adalet raporu basılacak; ayrıca `bias_comparison.png` grafiği oluşturulacaktır.*

### 4. FastAPI Sunucusunu Başlatma
Modeli web üzerinden erişilebilir kılmak için API backend sunucusunu çalıştırın:
```bash
uvicorn api.main:app --reload
```
*API ayağa kalktığında `http://localhost:8000` adresinden istek almaya hazır olacaktır.*

### 5. Simülatörü Açma
Sunucu çalışırken, `demo/index.html` dosyasını tarayıcınızda çift tıklayarak açabilirsiniz. Karşınıza tamamen çalışan, etkileşimli ve animasyonlu FairLoan simülatör kontrol paneli gelecektir.

---

## Teknolojiler & Kütüphaneler

- **Veri & Analiz:** Pandas, NumPy, Matplotlib
- **Makine Öğrenmesi (ML):** Scikit-Learn (Gradient Boosting Classifier, StandardScaler, Pipeline), Joblib
- **Backend API:** FastAPI, Pydantic, Uvicorn
- **Kullanıcı Arayüzü (UI):** Vanilla HTML5, CSS3 (Modern Glassmorphism & HSL Color Palettes), JavaScript (ES6+ Asynchronous Fetch & SVG Circular Progress Animations)
