import spacy
from spacy.training.example import Example
import random
import logging
from tqdm import tqdm # İlerleme çubuğu için

class ResumeNERTrainer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        # İngilizce metinler olduğu için İngilizce temelini kullanıyoruz
        self.nlp = spacy.blank("en") 
        self.ner = self.nlp.add_pipe("ner", last=True)

    def train(self, training_data, n_iter=10):
        """Modeli verilen veri ile eğitir."""
        
        # 1. Etiketleri modele tanıt
        logging.info("Etiketler modele ekleniyor...")
        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # 2. Eğitimi başlat
        optimizer = self.nlp.begin_training()
        
        logging.info(f"Eğitim başlıyor... ({n_iter} Epoch)")
        
        # 3. Epoch Döngüsü
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            
            # Batch'ler halinde değil, basitlik için tek tek (Example objesi ile) eğitiyoruz
            # (Büyük veri setlerinde minibatch tercih edilir ama bu proje için bu yeterli)
            for text, annotations in tqdm(training_data, desc=f"Epoch {itn+1}/{n_iter}"):
                try:
                    doc = self.nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    self.nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
                except Exception as e:
                    # Bazen karakter index hataları olabilir, onları atlayalım
                    pass
            
            logging.info(f"Epoch {itn+1} tamamlandı. Kayıp (Loss): {losses}")

    def save_model(self):
        """Eğitilen modeli diske kaydeder."""
        self.nlp.to_disk(self.output_dir)
        logging.info(f"Model başarıyla '{self.output_dir}' klasörüne kaydedildi.")