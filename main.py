import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_loader import ResumeDataLoader
from src.trainer import ResumeNERTrainer
from src.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_experiment(exp_name, base_model, train_data, test_data):
    """Tek bir deney (Eğitim + Test) çalıştırır."""
    print(f"\n{'='*50}\n EXPERIMENT STARTING: {exp_name}\n{'='*50}")
    
    output_dir = f"models/{exp_name.lower().replace(' ', '_')}"
    results_dir = f"results/{exp_name.lower().replace(' ', '_')}"
    
    # 1. Eğitim
    trainer = ResumeNERTrainer(output_dir, base_model=base_model)
    trainer.train(train_data, n_iter=15)
    trainer.save_model()
    
    # 2. Test
    evaluator = ModelEvaluator(output_dir, test_data, results_dir)
    df_metrics, f1_score = evaluator.evaluate()
    
    print(f"\n>>> {exp_name} F1 Score: {f1_score:.4f}")
    
    df_metrics['Model'] = exp_name
    return df_metrics

def plot_comparison(combined_df):
    """İki modelin sonuçlarını yan yana çizer."""
    if combined_df.empty: return

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    chart = sns.barplot(
        data=combined_df, 
        x="F1-Score", 
        y="Entity", 
        hue="Model", 
        palette="magma"
    )
    
    plt.title('Comparison: Blank Model vs. Pre-trained Model', fontsize=16, fontweight='bold')
    plt.xlabel('F1 Score', fontsize=12)
    plt.xlim(0, 1.1)
    plt.legend(title='Model Architecture')
    
    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    save_path = "results/final_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[INFO] Comparison plot saved: {save_path}")

def main():
    loader = ResumeDataLoader("data/raw/ner_resumes.json")
    train_data, test_data = loader.load_and_split_data()
    
    if not train_data: return

    experiments = [
        # Deney 1: Sıfırdan Model
        {"name": "Blank Model", "base": None},
        
        # Deney 2: Transfer Learning (Varsa)
        {"name": "Transfer Learning", "base": "en_core_web_lg"} 
    ]
    
    all_results = []

    # 3. Deneyleri Çalıştır
    for exp in experiments:
        df = run_experiment(exp["name"], exp["base"], train_data, test_data)
        all_results.append(df)
    
    # 4. Sonuçları Birleştir ve Kıyasla
    final_df = pd.concat(all_results)
    final_df.to_csv("results/comparison_table.csv", index=False)
    
    plot_comparison(final_df)
    print("\nPROJECT COMPLETED! 🚀")

if __name__ == "__main__":
    main()