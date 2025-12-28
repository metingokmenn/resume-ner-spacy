import json
import logging
import spacy
import re
from spacy.util import filter_spans
from spacy.training import offsets_to_biluo_tags
from sklearn.model_selection import train_test_split

class ResumeDataLoader:
    """
    JSON verilerini yükler; Skills (parçalama) ve Years of Experience (gelişmiş regex)
    için özel temizlik kuralları uygular.
    """

    def __init__(self, file_path, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.nlp_helper = spacy.blank("en")

    def _clean_entity_spans(self, text, start, end):
        """Genel temizlik (Trim)."""
        entity_text = text[start:end]
        # Parantezleri ve özel karakterleri de temizle
        chars_to_strip = ",.;:- \n\t\"'•()[]"

        if not entity_text.strip():
            return None, None

        while entity_text and entity_text[0] in chars_to_strip:
            start += 1
            entity_text = text[start:end]

        while entity_text and entity_text[-1] in chars_to_strip:
            end -= 1
            entity_text = text[start:end]

        if start >= end:
            return None, None

        return start, end

    def _refine_experience(self, text, start, end):
        """
        Sadece 'Years of Experience' için çok aşamalı kurtarma.
        """
        span_text = text[start:end]
        
        # 1. STRATEJİ: Tam Kalıp (Sayı + Birim + Opsiyonel Nokta)
        # Örn: "5+ Years", "5yrs.", "5-6 months", "3.5 Yrs"
        # yrs?\.? -> "yrs" veya "yrs." yı yakalar.
        pattern_strict = r'(\d+(?:[\.\-]\d+)?\+?)\s*(?:years?|yrs?\.?|months?|mnths?)'
        
        match = re.search(pattern_strict, span_text, re.IGNORECASE)
        
        if match:
            new_start = start + match.start()
            new_end = start + match.end()
            return new_start, new_end
            
        # 2. STRATEJİ: Sadece Sayı (Fallback)
        # Eğer "Experience: 5" yazıyorsa ve yıl yoksa, sadece "5"i alalım.
        # Bu hiç yoktan iyidir ve 0.0 puan almayı engeller.
        pattern_digits = r'(\d+(?:[\.\-]\d+)?\+?)'
        match_digit = re.search(pattern_digits, span_text)
        
        if match_digit:
             new_start = start + match_digit.start()
             new_end = start + match_digit.end()
             return new_start, new_end

        # 3. STRATEJİ: Hiçbiri olmazsa klasik temizlik yap
        return self._clean_entity_spans(text, start, end)

    def _split_skills(self, text, start, end, label):
        """Skills alanını virgüllere göre böler."""
        if label != "Skills":
            c_start, c_end = self._clean_entity_spans(text, start, end)
            if c_start is not None:
                return [(c_start, c_end, label)]
            return []

        span_text = text[start:end]
        new_entities = []
        matches = re.finditer(r'[^,;\n•]+', span_text)

        for match in matches:
            sub_start = start + match.start()
            sub_end = start + match.end()
            c_start, c_end = self._clean_entity_spans(text, sub_start, sub_end)
            
            # Çok kısa (1-2 karakter) skill'leri atla ama C, R, Go, AI gibileri tut
            if c_start and (c_end - c_start > 2 or text[c_start:c_end].strip().upper() in ['C', 'R', 'GO', 'AI', 'JS', 'ML']):
                new_entities.append((c_start, c_end, label))

        return new_entities

    def load_and_split_data(self):
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                raw_data = json.load(f)

            processed_data = []
            skipped_count = 0
            
            logging.info("Veri yükleme, Skills Split ve Gelişmiş Experience Regex işlemi başladı...")

            for data in raw_data:
                text = data.get('content', '')
                if not text or not data.get('annotation'):
                    continue

                doc = self.nlp_helper(text)
                spans = []
                
                for annot in data['annotation']:
                    if not annot.get('label') or not annot.get('points'):
                        continue
                        
                    label = annot['label'][0]
                    point = annot['points'][0]
                    start = point.get('start')
                    end = point.get('end')

                    if start is None or end is None: continue
                    try:
                        start = int(start)
                        end = int(end)
                    except ValueError: continue
                    
                    if start >= end: continue
                    end += 1 

                    # --- ETİKET ÖZEL İŞLEMLER ---
                    potential_entities = []
                    
                    if label == "Years of Experience":
                        e_start, e_end = self._refine_experience(text, start, end)
                        if e_start is not None:
                            potential_entities.append((e_start, e_end, label))
                    elif label == "Skills":
                        potential_entities = self._split_skills(text, start, end, label)
                    else:
                        c_start, c_end = self._clean_entity_spans(text, start, end)
                        if c_start is not None:
                            potential_entities.append((c_start, c_end, label))

                    # Oluşan tüm potansiyel etiketleri Spacy'ye hizala
                    for p_start, p_end, p_label in potential_entities:
                        # alignment_mode="contract": 
                        # Eğer "4yrs." metninde biz "4yrs" bulduysak ve Spacy "4", "yrs", "." diye böldüyse
                        # contract modu sadece "4" ve "yrs" tokenlarını alır, noktayı dışarıda bırakır.
                        span = doc.char_span(p_start, p_end, label=p_label, alignment_mode="contract")
                        if span:
                            spans.append(span)

                # Filtreleme ve Kayıt
                filtered_spans = filter_spans(spans)
                final_entities = [(span.start_char, span.end_char, span.label_) for span in filtered_spans]

                if final_entities:
                    try:
                        offsets_to_biluo_tags(doc, final_entities)
                        processed_data.append((text, {"entities": final_entities}))
                    except Exception:
                        skipped_count += 1
                        continue
            
            logging.info(f"İşlenemeyen Satır: {skipped_count}")
            
            if not processed_data:
                logging.error("Hiçbir veri işlenemedi!")
                return [], []

            train_data, test_data = train_test_split(
                processed_data, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            
            return train_data, test_data

        except Exception as e:
            logging.error(f"Veri yükleme hatası: {e}")
            return [], []