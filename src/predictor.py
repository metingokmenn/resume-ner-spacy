import spacy

class ResumeParser:
    def __init__(self, model_path):
        self.nlp = spacy.load(model_path)

    def get_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })
        return entities