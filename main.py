import logging
from src.data_loader import ResumeDataLoader
from src.trainer import ResumeNERTrainer
from src.predictor import ResumeParser

# Loglama ayarları
logging.basicConfig(level=logging.INFO)

def main():
    # --- AYARLAR ---
    DATA_PATH = "data/raw/ner_resumes.json" # Dosya ismini kontrol et!
    MODEL_OUTPUT_DIR = "models/output_model"
    EPOCH_SAYISI = 10

    # 1. Veriyi Yükle
    print("--- 1. ADIM: Veri Yükleniyor ---")
    loader = ResumeDataLoader(DATA_PATH)
    train_data = loader.load_data()

    # Verinin %90'ını eğitim, %10'unu test gibi düşünelim (Burada hepsini eğitime sokuyoruz şimdilik)
    if not train_data:
        print("Veri bulunamadı, işlem durduruluyor.")
        return

    # 2. Modeli Eğit
    print(f"\n--- 2. ADIM: Model Eğitiliyor ({len(train_data)} veri ile) ---")
    trainer = ResumeNERTrainer(MODEL_OUTPUT_DIR)
    trainer.train(train_data, n_iter=EPOCH_SAYISI)
    trainer.save_model()

    # 3. Test Et (Kendi verinle dene)
    print("\n--- 3. ADIM: Test Ediliyor ---")
    parser = ResumeParser(MODEL_OUTPUT_DIR)
    
    test_metni = """
    Metin Ozturk
    Software Engineer
    Skills: Python, Java, Machine Learning, SQL
    Experience: 
    Google - Senior Developer (2020 - 2023)
    """
    
    sonuclar = parser.get_entities(test_metni)
    
    print("\n--- BULUNAN VARLIKLAR ---")
    for item in sonuclar:
        print(f"[{item['label']}] -> {item['text']}")

if __name__ == "__main__":
    main()