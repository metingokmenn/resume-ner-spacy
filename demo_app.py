import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import spacy
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ResumeNERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume NER Intelligence - Demo (Hybrid vs Pure ML Analysis)")
        self.root.geometry("1300x850")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Veri Değişkenleri
        self.nlp = None
        self.resume_data = []
        self.current_index = 0
        self.current_model_name = "Unknown Model"
        
        # İstatistik Değişkenleri
        self.results_map = {}
        
        # Pure ML İstatistikleri
        self.total_processed = 0
        self.total_hybrid_acc_sum = 0.0
        self.total_ml_acc_sum = 0.0
        
        # Renk Haritası
        self.entity_colors = {
            "Skills": "#ffcccb", "Name": "#ccffcc", "College Name": "#cce5ff",
            "Degree": "#e5ccff", "Years of Experience": "#ffffcc", 
            "Email Address": "#ffe5cc", "Companies worked at": "#d9d9d9"
        }

        self._create_widgets()

    def _create_widgets(self):
        # --- ÜST PANEL ---
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED, bg="#e0e0e0")
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Butonlar
        btn_load_model = tk.Button(toolbar, text="📂 1. Modeli Yükle", command=self.load_model, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_load_model.pack(side=tk.LEFT, padx=5, pady=5)

        btn_load_json = tk.Button(toolbar, text="📄 2. Test Verisi Yükle", command=self.load_json, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        btn_load_json.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_status = tk.Label(toolbar, text="Durum: Bekleniyor...", fg="red", bg="#e0e0e0", font=("Arial", 10))
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        # --- SKOR PANELI ---
        self.score_frame = tk.Frame(toolbar, bg="#f5f5f5", bd=2, relief=tk.GROOVE)
        self.score_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Satır 1: Hybrid Model
        lbl_hybrid_title = tk.Label(self.score_frame, text="HYBRID MODEL:", font=("Arial", 9, "bold"), fg="#2E7D32", bg="#f5f5f5")
        lbl_hybrid_title.grid(row=0, column=0, sticky="w", padx=5)
        
        self.lbl_hybrid_curr = tk.Label(self.score_frame, text="Anlık: -", font=("Arial", 10, "bold"), fg="#333", bg="#f5f5f5")
        self.lbl_hybrid_curr.grid(row=0, column=1, padx=5)
        
        self.lbl_hybrid_avg = tk.Label(self.score_frame, text="Ort: -", font=("Arial", 10, "bold"), fg="#333", bg="#f5f5f5")
        self.lbl_hybrid_avg.grid(row=0, column=2, padx=5)

        # Satır 2: Pure ML
        lbl_ml_title = tk.Label(self.score_frame, text="PURE ML (No Rules):", font=("Arial", 9, "bold"), fg="#D32F2F", bg="#f5f5f5")
        lbl_ml_title.grid(row=1, column=0, sticky="w", padx=5)
        
        self.lbl_ml_curr = tk.Label(self.score_frame, text="Anlık: -", font=("Arial", 10), fg="#666", bg="#f5f5f5")
        self.lbl_ml_curr.grid(row=1, column=1, padx=5)
        
        self.lbl_ml_avg = tk.Label(self.score_frame, text="Ort: -", font=("Arial", 10), fg="#666", bg="#f5f5f5")
        self.lbl_ml_avg.grid(row=1, column=2, padx=5)

        # --- ANA ALAN ---
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # SOL: Metin
        left_frame = ttk.LabelFrame(paned_window, text="Özgeçmiş İçeriği (Raw Text)")
        paned_window.add(left_frame, weight=3)
        
        self.text_area = tk.Text(left_frame, wrap=tk.WORD, font=("Consolas", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for label, color in self.entity_colors.items():
            self.text_area.tag_config(label, background=color)

        # SAĞ: Tablo
        right_frame = ttk.LabelFrame(paned_window, text="Çıkarılan Varlıklar (Hybrid Results)")
        paned_window.add(right_frame, weight=2)

        columns = ("Entity Text", "Label", "Status")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings")
        self.tree.heading("Entity Text", text="Değer")
        self.tree.heading("Label", text="Etiket")
        self.tree.heading("Status", text="Durum")
        
        self.tree.column("Entity Text", width=150)
        self.tree.column("Label", width=120)
        self.tree.column("Status", width=60, anchor="center")
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- ALT PANEL ---
        nav_frame = tk.Frame(self.root, bg="#f0f0f0")
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        btn_prev = tk.Button(nav_frame, text="<< Önceki", command=self.prev_resume)
        btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.lbl_counter = tk.Label(nav_frame, text="0 / 0", font=("Arial", 10), bg="#f0f0f0")
        self.lbl_counter.pack(side=tk.LEFT, padx=20)

        btn_next = tk.Button(nav_frame, text="Sonraki >>", command=self.next_resume)
        btn_next.pack(side=tk.LEFT, padx=10)
        
        btn_process = tk.Button(nav_frame, text="⚙️ ANALİZ ET (Hybrid vs Pure ML)", command=self.process_current_resume, bg="#FF5722", fg="white", font=("Arial", 11, "bold"))
        btn_process.pack(side=tk.RIGHT, padx=10)

        btn_report = tk.Button(nav_frame, text="📊 SONUÇ GRAFİĞİ", command=self.generate_inference_report, bg="#673AB7", fg="white", font=("Arial", 11, "bold"))
        btn_report.pack(side=tk.RIGHT, padx=10)

    def load_model(self):
        initial_dir = "models"
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()
            
        path = filedialog.askdirectory(initialdir=initial_dir, title="Model Klasörünü Seç")
        
        if path:
            try:
                folder_name = os.path.basename(path)
                self.current_model_name = folder_name.replace("_", " ").title()
                
                self.nlp = spacy.load(path)
                
                # Hibrit Katman (Rule-Based)
                if "entity_ruler" not in self.nlp.pipe_names:
                    ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                    patterns = [
                        {"label": "Years of Experience", "pattern": [{"TEXT": {"REGEX": r"^\d+(\.\d+)?\+?$"}}, {"LOWER": {"IN": ["year", "years", "yr", "yrs", "month", "months", "mnths"]}}]},
                        {"label": "Years of Experience", "pattern": [{"TEXT": {"REGEX": r"^\d+$"}}, {"TEXT": "+"}, {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}}]},
                        {"label": "Years of Experience", "pattern": [{"LOWER": "one"}, {"LOWER": {"IN": ["year", "yr"]}}]},
                        {"label": "Years of Experience", "pattern": [{"LOWER": "two"}, {"LOWER": {"IN": ["years", "yrs"]}}]}
                    ]
                    ruler.add_patterns(patterns)
                    print("✅ Hibrit Katman (Kod İçi) Başarıyla Eklendi.")
                    self.lbl_status.config(text=f"Model: {self.current_model_name} + Rule-Based", fg="green")
                else:
                    self.lbl_status.config(text=f"Model: {self.current_model_name} (Dahili Ruler)", fg="green")

                messagebox.showinfo("Bilgi", f"Model Başarıyla Yüklendi:\n{self.current_model_name}")
                
            except Exception as e:
                print(f"❌ KRİTİK HATA: {e}")
                self.lbl_status.config(text="Model Yükleme Hatası!", fg="red")
                messagebox.showerror("Hata", f"Model yüklenirken hata oluştu:\n{str(e)}")

    def load_json(self):
        initial_dir = os.path.join("data", "processed")
        if not os.path.exists(initial_dir): initial_dir = os.getcwd()
        path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("JSON", "*.json")])
        if path:
            try:
                with open(path, 'r', encoding="utf-8") as f:
                    self.resume_data = json.load(f)
                self.current_index = 0
                self.results_map = {} 
                self.total_processed = 0
                self.total_hybrid_acc_sum = 0.0
                self.total_ml_acc_sum = 0.0
                
                self.update_display()
                self.lbl_status.config(text="Veri Yüklendi. Analiz Bekleniyor.", fg="orange")
                self._reset_labels()
            except Exception as e:
                messagebox.showerror("Hata", str(e))

    def _reset_labels(self):
        self.lbl_hybrid_curr.config(text="Anlık: -")
        self.lbl_hybrid_avg.config(text="Ort: -")
        self.lbl_ml_curr.config(text="Anlık: -")
        self.lbl_ml_avg.config(text="Ort: -")

    def update_display(self):
        if not self.resume_data: return
        self.text_area.delete("1.0", tk.END)
        current_content = self.resume_data[self.current_index].get('content', '')
        self.text_area.insert(tk.END, current_content)
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_counter.config(text=f"{self.current_index + 1} / {len(self.resume_data)}")

    def _calculate_accuracy(self, doc, ground_truth_set):
        hits = 0
        for ent in doc.ents:
            if (ent.label_, ent.text.strip()) in ground_truth_set:
                hits += 1
        
        total_expected = len(ground_truth_set)
        if total_expected > 0:
            return (hits / total_expected) * 100
        else:
            return 100 if hits == 0 else 0

    def process_current_resume(self):
        if not self.nlp or not self.resume_data: 
            messagebox.showwarning("Uyarı", "Model veya Veri eksik!")
            return

        text = self.text_area.get("1.0", tk.END).strip()
        
        # 1. Ground Truth
        current_annotations = self.resume_data[self.current_index].get("annotation", [])
        ground_truth_set = set()
        for ann in current_annotations:
            lbl = ann["label"][0]
            points = ann["points"][0]
            truth_text = points.get("text", text[points["start"]:points["end"]+1]).strip()
            ground_truth_set.add((lbl, truth_text))

        # 2. Hybrid Run
        doc_hybrid = self.nlp(text)
        acc_hybrid = self._calculate_accuracy(doc_hybrid, ground_truth_set)

        # 3. Pure ML Run
        if "entity_ruler" in self.nlp.pipe_names:
            with self.nlp.disable_pipes("entity_ruler"):
                doc_ml = self.nlp(text)
        else:
            doc_ml = doc_hybrid
        acc_ml = self._calculate_accuracy(doc_ml, ground_truth_set)

        # 4. Tablo & Highlight
        predictions = []
        for ent in doc_hybrid.ents:
            start_idx = "1.0"
            while True:
                start_idx = self.text_area.search(ent.text, start_idx, stopindex=tk.END)
                if not start_idx: break
                end_idx = f"{start_idx}+{len(ent.text)}c"
                self.text_area.tag_add(ent.label_, start_idx, end_idx)
                start_idx = end_idx
            
            is_correct = (ent.label_, ent.text.strip()) in ground_truth_set
            status_icon = "✅" if is_correct else "⚠️"
            predictions.append((ent.text, ent.label_, status_icon))

        for i in self.tree.get_children(): self.tree.delete(i)
        for pred in predictions:
            self.tree.insert("", tk.END, values=pred)

        # 5. İstatistikler
        self.total_processed += 1
        self.total_hybrid_acc_sum += acc_hybrid
        self.total_ml_acc_sum += acc_ml
        
        avg_hybrid = self.total_hybrid_acc_sum / self.total_processed
        avg_ml = self.total_ml_acc_sum / self.total_processed
        
        self.results_map[self.current_index] = acc_hybrid

        # 6. Etiketler
        self.lbl_hybrid_curr.config(text=f"Anlık: %{acc_hybrid:.1f}", fg="green" if acc_hybrid > 70 else "orange")
        self.lbl_hybrid_avg.config(text=f"Ort: %{avg_hybrid:.1f}")
        
        self.lbl_ml_curr.config(text=f"Anlık: %{acc_ml:.1f}", fg="red" if acc_ml < acc_hybrid else "gray")
        self.lbl_ml_avg.config(text=f"Ort: %{avg_ml:.1f}")

        self.lbl_status.config(text="Analiz Tamamlandı", fg="blue")

    def generate_inference_report(self):
        if not self.results_map:
            messagebox.showwarning("Uyarı", "Henüz hiçbir CV analiz edilmedi.")
            return

        sorted_indices = sorted(self.results_map.keys())
        x_labels = [f"CV-{i+1}" for i in sorted_indices]
        y_scores = [self.results_map[i] for i in sorted_indices]
        avg_score = sum(y_scores) / len(y_scores)

        # --- YENİ: KAYDETME VE GÖSTERME MANTIĞI ---
        
        # Grafik Oluştur
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        
        bars = ax.bar(x_labels, y_scores, color='#4CAF50', alpha=0.7, label='Hybrid Accuracy')
        ax.axhline(y=avg_score, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_score:.1f}%')

        ax.set_ylim(0, 110)
        ax.set_ylabel('Accuracy (%)')
        
        ax.set_title(f'{self.current_model_name}\nInference Performance Report', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=9)
        
        # --- DOSYA KAYDETME ---
        # 1. results klasörü yoksa oluştur
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
           
        # 2. Model ismine göre klasör oluştur (örn: results/Transfer Learning)
        current_model_name_folder = "blank_model" if self.current_model_name == "Blank Model" else "transfer_learning"
        model_save_dir = os.path.join(results_dir, current_model_name_folder)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            
        # 3. Dosyayı kaydet
        save_path = os.path.join(model_save_dir, "inference_report_" + current_model_name_folder + ".png")
        try:
            fig.savefig(save_path)
            print(f"✅ Grafik kaydedildi: {save_path}")
            messagebox.showinfo("Başarılı", f"Grafik kaydedildi:\n{save_path}")
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")
            messagebox.showerror("Hata", f"Grafik kaydedilemedi:\n{e}")

        # --- EKRANA GÖSTERME (PENCERE) ---
        report_win = tk.Toplevel(self.root)
        report_win.title("Inference Performance Report")
        report_win.geometry("800x600")
        
        canvas = FigureCanvasTkAgg(fig, master=report_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def next_resume(self):
        if self.resume_data and self.current_index < len(self.resume_data) - 1:
            self.current_index += 1
            self.update_display()

    def prev_resume(self):
        if self.resume_data and self.current_index > 0:
            self.current_index -= 1
            self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeNERApp(root)
    root.mainloop()