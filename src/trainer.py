import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import logging
import os
from tqdm import tqdm

class ResumeNERTrainer:
    """
    Spacy tabanlı Named Entity Recognition modelini eğitir.
    """

    def __init__(self, output_dir, lang="en"):
        """
        Args:
            output_dir (str): Modelin kaydedileceği dizin.
            lang (str): Model dili (default: 'en').
        """
        self.output_dir = output_dir
        self.nlp = spacy.blank(lang)
        self.ner = self.nlp.add_pipe("ner", last=True)

    def train(self, training_data, n_iter=15):
        """
        Modeli verilen veri seti üzerinde eğitir.
        Optimizasyon: Compounding batch size ve Dropout kullanır.
        
        Args:
            training_data (list): Spacy formatında eğitim verisi.
            n_iter (int): Epoch sayısı.
        """
        # Etiketleri ekle
        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Eğitimi başlat
        optimizer = self.nlp.begin_training()
        
        # Batch boyutunu dinamik olarak ayarla (Performans artırıcı)
        sizes = compounding(4.0, 32.0, 1.001)
        
        logging.info(f"Eğitim başlıyor... ({n_iter} Epoch)")
        
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=sizes)
            
            # Batch'ler halinde eğitim
            for batch in batches:
                texts, annotations = zip(*batch)
                example_batch = []
                for i in range(len(texts)):
                    try:
                        doc = self.nlp.make_doc(texts[i])
                        example = Example.from_dict(doc, annotations[i])
                        example_batch.append(example)
                    except Exception:
                        continue
                
                # Dropout: Overfitting'i engeller (0.5 -> 0.2 arası deneyimsel optimum)
                self.nlp.update(example_batch, drop=0.35, sgd=optimizer, losses=losses)
            
            logging.info(f"Epoch {itn+1}/{n_iter} - Loss: {losses['ner']:.2f}")

    def save_model(self):
        """Eğitilen modeli diske kaydeder."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.nlp.to_disk(self.output_dir)
        logging.info(f"Model başarıyla '{self.output_dir}' konumuna kaydedildi.")