import spacy
from spacy.training.example import Example
from spacy.scorer import Scorer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

class ModelEvaluator:
    """
    Eğitilen modeli test verisi üzerinde değerlendirir ve raporlar oluşturur.
    """

    def __init__(self, model_path, test_data, results_dir="results"):
        self.nlp = spacy.load(model_path)
        self.test_data = test_data
        self.results_dir = results_dir
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def evaluate(self):
        """
        Test verisi üzerinde metrikleri (Precision, Recall, F1) hesaplar.
        Returns:
            pd.DataFrame: Sonuç tablosu.
        """
        logging.info("Değerlendirme süreci başladı...")
        scorer = Scorer()
        examples = []
        
        for text, annotations in self.test_data:
            doc = self.nlp.make_doc(text)
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
        
        # CSV olarak kaydet
        csv_path = os.path.join(self.results_dir, "evaluation_metrics.csv")
        df.to_csv(csv_path, index=False)
        
        # Grafiği çiz
        self._plot_results(df)
        
        logging.info(f"Genel Model Başarısı (F1): {scores.get('ents_f', 0):.4f}")
        logging.info(f"Raporlar '{self.results_dir}' klasörüne kaydedildi.")
        return df

    def _plot_results(self, df):
        """F1 skorlarını görselleştirir ve kaydeder."""
        if df.empty:
            return

        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        barplot = sns.barplot(x="F1-Score", y="Entity", data=df, palette="viridis", hue="Entity", legend=False)
        
        plt.title('Varlık Tanıma Başarısı (Test Seti)', fontsize=16)
        plt.xlabel('F1 Score', fontsize=12)
        plt.ylabel('Etiket Tipi', fontsize=12)
        plt.xlim(0, 1.1)
        
        for i in barplot.containers:
            barplot.bar_label(i, fmt='%.2f', padding=3)

        save_path = os.path.join(self.results_dir, "f1_score_chart.png")
        plt.tight_layout()
        plt.savefig(save_path)