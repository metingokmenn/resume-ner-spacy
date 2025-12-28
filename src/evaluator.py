import spacy
from spacy.training.example import Example
from spacy.scorer import Scorer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

class ModelEvaluator:
    def __init__(self, model_path, test_data, results_dir="results"):
        self.nlp = spacy.load(model_path)
        self.test_data = test_data
        self.results_dir = results_dir
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def evaluate(self):
        logging.info("Değerlendirme başladı...")
        scorer = Scorer()
        examples = []
        
        for text, annotations in self.test_data:
            doc = self.nlp(text) # Tahmin yap
            try:
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            except Exception:
                pass

        scores = scorer.score(examples)
        ents_per_type = scores.get('ents_per_type', {})
        
        data = []
        for label, metrics in ents_per_type.items():
            data.append({
                'Entity': label,
                'Precision': metrics['p'],
                'Recall': metrics['r'],
                'F1-Score': metrics['f']
            })
            
        df = pd.DataFrame(data).sort_values(by='F1-Score', ascending=False)
        
        csv_path = os.path.join(self.results_dir, "metrics.csv")
        df.to_csv(csv_path, index=False)
        
        self._plot_results(df)
        return df, scores.get('ents_f', 0)

    def _plot_results(self, df):
        if df.empty: return

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        barplot = sns.barplot(x="F1-Score", y="Entity", data=df, palette="viridis", hue="Entity", legend=False)
        plt.title('Varlık Tanıma Başarısı', fontsize=14)
        plt.xlim(0, 1.1)
        
        for i in barplot.containers:
            barplot.bar_label(i, fmt='%.2f', padding=3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "f1_chart.png"))
        plt.close() # Hafızayı temizle