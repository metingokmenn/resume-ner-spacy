import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random
import logging
import os
from tqdm import tqdm

class ResumeNERTrainer:
    def __init__(self, output_dir, base_model=None):
        """
        Args:
            output_dir (str): Modelin kaydedileceği yer.
            base_model (str): None ise sıfırdan, 'en_core_web_lg' ise transfer learning.
        """
        self.output_dir = output_dir
        self.base_model_name = base_model

        if base_model:
            logging.info(f"Transfer Learning başlatılıyor: '{base_model}' yükleniyor...")
            try:
                self.nlp = spacy.load(base_model)
            except OSError:
                logging.warning(f"MODEL BULUNAMADI: '{base_model}'. Lütfen 'python -m spacy download {base_model}' çalıştırın.")
                logging.warning("Otomatik olarak 'Blank Model'e geçiliyor.")
                self.nlp = spacy.blank("en")
        else:
            logging.info("Sıfırdan eğitim (Blank Model) başlatılıyor...")
            self.nlp = spacy.blank("en")

        # NER pipe kontrolü
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner", last=True)
        else:
            self.ner = self.nlp.get_pipe("ner")

    def train(self, training_data, n_iter=15):
        # 1. Etiketleri Modele Ekle
        for _, annotations in training_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # 2. Optimizer Ayarı
        # Eğer pre-trained modelse, ağırlıkları sıfırlama (resume_training veya create_optimizer)
        # Eğer blank modelse, ağırlıkları başlat (begin_training)
        if self.base_model_name:
            optimizer = self.nlp.create_optimizer()
        else:
            optimizer = self.nlp.begin_training()

        # 3. Eğitim Döngüsü
        sizes = compounding(4.0, 32.0, 1.001)
        
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=sizes)
            
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
                
                try:
                    self.nlp.update(example_batch, drop=0.35, sgd=optimizer, losses=losses)
                except Exception:
                    pass
            
            logging.info(f"Epoch {itn+1}/{n_iter} - Loss: {losses.get('ner', 0):.2f}")

    def save_model(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.nlp.to_disk(self.output_dir)