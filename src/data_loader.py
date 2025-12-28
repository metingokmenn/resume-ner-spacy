import json
import logging
from sklearn.model_selection import train_test_split

class ResumeDataLoader:
    """
    JSON formatındaki özgeçmiş verilerini yükler, temizler ve
    Eğitim/Test setlerine ayırır.
    """

    def __init__(self, file_path, test_size=0.2, random_state=42):
        """
        Args:
            file_path (str): JSON dosyasının yolu.
            test_size (float): Test verisi oranı (0.0 - 1.0 arası).
            random_state (int): Tekrarlanabilirlik için seed değeri.
        """
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state

    def _trim_entity_spans(self, text, start, end):
        """
        Etiket sınırlarını (span) kontrol ederek baştaki/sondaki
        gereksiz boşlukları ve karakterleri temizler.
        """
        entity_text = text[start:end]
        if not entity_text.strip():
            return None, None

        while entity_text and (entity_text.startswith(" ") or entity_text.startswith("\n")):
            start += 1
            entity_text = text[start:end]

        while entity_text and (entity_text.endswith(" ") or entity_text.endswith("\n")):
            end -= 1
            entity_text = text[start:end]

        return start, end

    def load_and_split_data(self):
        """
        Veriyi yükler, temizler ve train/test olarak ayırır.

        Returns:
            tuple: (train_data, test_data) formatında Spacy uyumlu listeler.
        """
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                raw_data = json.load(f)

            processed_data = []
            
            for data in raw_data:
                text = data.get('content', '')
                entities = []
                
                if not text or not data.get('annotation'):
                    continue

                for annot in data['annotation']:
                    if not annot.get('label') or not annot.get('points'):
                        continue
                        
                    label = annot['label'][0]
                    point = annot['points'][0]
                    start = point.get('start')
                    end = point.get('end')

                    if start is None or end is None:
                        continue

                    try:
                        start = int(start)
                        end = int(end)
                    except ValueError:
                        continue

                    if start >= end:
                        continue
                    
                    # Dataturks formatı düzeltmesi ve trim işlemi
                    end += 1
                    clean_start, clean_end = self._trim_entity_spans(text, start, end)
                    
                    if clean_start is None or clean_start >= clean_end:
                        continue

                    entities.append((clean_start, clean_end, label))

                if entities:
                    processed_data.append((text, {"entities": entities}))

            # Akademik Standart: Random Split
            train_data, test_data = train_test_split(
                processed_data, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            
            logging.info(f"Veri işlendi. Train: {len(train_data)}, Test: {len(test_data)}")
            return train_data, test_data

        except Exception as e:
            logging.error(f"Veri işleme hatası: {e}")
            return [], []