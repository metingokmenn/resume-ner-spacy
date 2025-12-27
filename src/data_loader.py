import json
import logging

class ResumeDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.training_data = []

    def load_data(self):
        """Dataturks JSON formatındaki veriyi okur ve Spacy formatına çevirir."""
        try:
            with open(self.file_path, 'r', encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data['content']
                entities = []
                
                if data['annotation'] is None:
                    continue

                for annotation in data['annotation']:
                    # Etiket ismini al
                    label = annotation['label'][0]
                    
                    # Koordinatları al
                    point = annotation['points'][0]
                    start = point['start']
                    end = point['end']

                    # Dataturks formatında 'end' indexi dahildir, Spacy hariç tutar.
                    # Bu yüzden +1 ekliyoruz. Ayrıca etiketleri temizliyoruz.
                    if start < end:
                         entities.append((start, end + 1, label))

                self.training_data.append((text, {"entities": entities}))
            
            logging.info(f"{len(self.training_data)} adet veri başarıyla yüklendi ve dönüştürüldü.")
            return self.training_data

        except Exception as e:
            logging.error(f"Veri yüklenirken hata oluştu: {e}")
            return []