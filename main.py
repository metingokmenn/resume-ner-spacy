import logging
from src.data_loader import ResumeDataLoader
from src.trainer import ResumeNERTrainer
from src.evaluator import ModelEvaluator

# Loglama konfigürasyonu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- PROJE AYARLARI ---
    DATA_PATH = "data/raw/ner_resumes.json"
    MODEL_OUTPUT_DIR = "models/resume_ner_model"
    EPOCH = 15  # Akademik deneylerde genelde 15-20 epoch ile başlanır

    # 1. Veri Hazırlığı
    print("\n" + "="*40)
    print(" 1. AŞAMA: Veri Hazırlığı ve Ayrıştırma")
    print("="*40)
    loader = ResumeDataLoader(DATA_PATH)
    train_data, test_data = loader.load_and_split_data()

    if not train_data:
        logging.error("Veri seti boş veya yüklenemedi.")
        return

    # 2. Model Eğitimi
    print("\n" + "="*40)
    print(f" 2. AŞAMA: Model Eğitimi (Train Size: {len(train_data)})")
    print("="*40)
    trainer = ResumeNERTrainer(MODEL_OUTPUT_DIR)
    trainer.train(train_data, n_iter=EPOCH)
    trainer.save_model()

    # 3. Değerlendirme
    print("\n" + "="*40)
    print(f" 3. AŞAMA: Performans Ölçümü (Test Size: {len(test_data)})")
    print("="*40)
    evaluator = ModelEvaluator(MODEL_OUTPUT_DIR, test_data)
    results = evaluator.evaluate()
    
    print("\n--- ÖZET SONUÇ TABLOSU ---")
    print(results.head(10))  # İlk 10 sonucu göster

if __name__ == "__main__":
    main()