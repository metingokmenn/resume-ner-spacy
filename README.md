# Information Extraction from Unstructured Resumes using NER

## 📌 Proje Özeti (Abstract)

Bu proje, yapısal olmayan (unstructured) özgeçmiş metinlerinden yapılandırılmış bilgi (İsim, Yetenekler, Okul, Deneyim vb.) çıkarmayı amaçlayan bir Doğal Dil İşleme (NLP) çalışmasıdır. Proje kapsamında, Spacy kütüphanesi kullanılarak özel bir Varlık Tanıma (Named Entity Recognition - NER) modeli eğitilmiş ve test edilmiştir.

## 🛠️ Yöntem (Methodology)

Proje aşağıdaki akademik boru hattını (pipeline) takip etmektedir:

1.  **Veri Toplama:** Kaggle kaynaklı 220 adet etiketlenmiş özgeçmiş verisi kullanıldı.
2.  **Ön İşleme (Preprocessing):**
    - Veri temizliği (Boşluk ve karakter düzeltmeleri).
    - Alignment (Hizalama) sorunları için özel `Span Trimming` algoritması geliştirildi.
3.  **Veri Bölümleme:** Veri seti, modelin genelleme yeteneğini ölçmek amacıyla **%80 Eğitim (Train)** ve **%20 Test** olarak randomize şekilde ayrıldı.
4.  **Model Eğitimi:**
    - **Mimari:** Transition-based NER (Spacy).
    - **Optimizasyon:** `Compounding Batch Size` ve `Dropout Decay` teknikleri ile overfitting engellendi.
5.  **Değerlendirme:** Test seti üzerinde Precision, Recall ve F1-Score metrikleri hesaplandı.

## 📂 Proje Yapısı

- `data/`: Ham veri setleri.
- `src/`: Kaynak kodlar (Loader, Trainer, Evaluator).
- `models/`: Eğitilmiş model çıktıları.
- `results/`: Performans grafikleri ve metrik tabloları.

## 📊 Deneysel Sonuçlar

Modelin test veri seti üzerindeki başarısı `results/evaluation_metrics.csv` dosyasında detaylandırılmıştır. Genel F1 skoru ve etiket bazlı başarı dağılımı `results/f1_score_chart.png` grafiğinde sunulmuştur.

## 🚀 Kurulum ve Çalıştırma

1. **Gereksinimleri Yükleyin:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Projeyi Başlatın:**

   ```bash
   python main.py
   ```

   Bu komut veri işleme, eğitim ve test süreçlerini otomatik olarak sırayla çalıştırır.

---

## Geliştirici

**Ad Soyad:** Metin Gökmen

**Ders:** Doğal Dil İşlemeye Kavramsal Bir Bakış

### 🚫 `.gitignore`

Git reposunu temiz tutmak için:

```text
# Python sanal ortam
venv/
__pycache__/
*.pyc

# Model dosyaları (Büyük olabilir)
models/

# Sonuçlar (Tekrar üretilebilir)
results/

# Sistem dosyaları
.DS_Store
```
