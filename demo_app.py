import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import spacy
import os

class ResumeNERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume NER Intelligence - Demo (Hybrid vs. Pure ML)")
        self.root.geometry("1280x800")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Veri Değişkenleri
        self.nlp = None
        self.resume_data = []
        self.current_index = 0
        
        # İstatistik Değişkenleri
        self.total_processed = 0
        self.total_hybrid_acc = 0.0
        self.total_pure_acc = 0.0
        
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
        btn_frame = tk.Frame(toolbar, bg="#e0e0e0")
        btn_frame.pack(side=tk.LEFT)
        
        btn_load_model = tk.Button(btn_frame, text="📂 1. Modeli Yükle", command=self.load_model, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_load_model.pack(side=tk.LEFT, padx=5, pady=5)

        btn_load_json = tk.Button(btn_frame, text="📄 2. Test Verisi Yükle", command=self.load_json, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        btn_load_json.pack(side=tk.LEFT, padx=5, pady=5)

        self.lbl_status = tk.Label(btn_frame, text="Durum: Bekleniyor...", fg="red", bg="#e0e0e0", font=("Arial", 10))
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        # --- SKOR PANELI (GELİŞMİŞ) ---
        self.score_frame = tk.Frame(toolbar, bg="white", bd=2, relief=tk.SUNKEN)
        self.score_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Grid Layout kullanalım düzgün hizalama için
        tk.Label(self.score_frame, text="MODEL PERFORMANSI", font=("Arial", 8, "bold"), bg="white", fg="gray").grid(row=0, column=0, columnspan=3, pady=2)
        
        # Başlıklar
        tk.Label(self.score_frame, text="Anlık", font=("Arial", 8, "bold"), bg="white", fg="gray").grid(row=1, column=1, padx=5)
        tk.Label(self.score_frame, text="Genel", font=("Arial", 8, "bold"), bg="white", fg="gray").grid(row=1, column=2, padx=5)
        
        # Satır 1: Hibrit Model (Bizim önerdiğimiz)
        tk.Label(self.score_frame, text="HİBRİT:", font=("Arial", 10, "bold"), bg="white", anchor="w").grid(row=2, column=0, padx=5, sticky="w")
        self.lbl_hybrid_score = tk.Label(self.score_frame, text="-%", font=("Arial", 12, "bold"), bg="white", fg="#2E7D32") # Koyu Yeşil
        self.lbl_hybrid_score.grid(row=2, column=1, padx=5, sticky="e")
        
        self.lbl_hybrid_total = tk.Label(self.score_frame, text="-%", font=("Arial", 10, "bold"), bg="white", fg="#2E7D32")
        self.lbl_hybrid_total.grid(row=2, column=2, padx=5, sticky="e")
        
        # Satır 2: Saf ML Model (Kıyaslama)
        tk.Label(self.score_frame, text="SAF ML:", font=("Arial", 10), bg="white", anchor="w").grid(row=3, column=0, padx=5, sticky="w")
        self.lbl_pure_score = tk.Label(self.score_frame, text="-%", font=("Arial", 11), bg="white", fg="#D32F2F") # Koyu Kırmızı
        self.lbl_pure_score.grid(row=3, column=1, padx=5, sticky="e")
        
        self.lbl_pure_total = tk.Label(self.score_frame, text="-%", font=("Arial", 10), bg="white", fg="#D32F2F")
        self.lbl_pure_total.grid(row=3, column=2, padx=5, sticky="e")

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
        right_frame = ttk.LabelFrame(paned_window, text="Çıkarılan Varlıklar (Hibrit Sonuç)")
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
        
        btn_process = tk.Button(nav_frame, text="⚙️ ANALİZ ET & KARŞILAŞTIR", command=self.process_current_resume, bg="#FF5722", fg="white", font=("Arial", 11, "bold"))
        btn_process.pack(side=tk.RIGHT, padx=10)

    def load_model(self):
        initial_dir = "models"
        if not os.path.exists(initial_dir): initial_dir = os.getcwd()
            
        path = filedialog.askdirectory(initialdir=initial_dir, title="Model Klasörünü Seç")
        
        if path:
            try:
                # 1. Modeli Yükle
                self.nlp = spacy.load(path)
                
                # 2. Hibrit Katman (Rule-Based) - KOD İÇİNDE GÖMÜLÜ
                if "entity_ruler" not in self.nlp.pipe_names:
                    ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                    
                    patterns = [
                        # Experience Patterns
                        {"label": "Years of Experience", "pattern": [{"TEXT": {"REGEX": r"^\d+(\.\d+)?\+?$"}}, {"LOWER": {"IN": ["year", "years", "yr", "yrs", "month", "months"]}}]},
                        {"label": "Years of Experience", "pattern": [{"TEXT": {"REGEX": r"^\d+$"}}, {"TEXT": "+"}, {"LOWER": {"IN": ["year", "years", "yr", "yrs"]}}]},
                        {"label": "Years of Experience", "pattern": [{"LOWER": "one"}, {"LOWER": {"IN": ["year", "yr"]}}]},
                        {"label": "Years of Experience", "pattern": [{"LOWER": "two"}, {"LOWER": {"IN": ["years", "yrs"]}}]}
                    ]
                    ruler.add_patterns(patterns)
                    print("✅ Hibrit Katman Eklendi.")
                    self.lbl_status.config(text="Model + Rule-Based Aktif", fg="green")
                else:
                    self.lbl_status.config(text="Model Yüklendi (Dahili Ruler)", fg="green")

                messagebox.showinfo("Bilgi", f"Model Başarıyla Yüklendi:\n{os.path.basename(path)}")
                
            except Exception as e:
                self.lbl_status.config(text="Model Hatası!", fg="red")
                messagebox.showerror("Hata", str(e))

    def load_json(self):
        initial_dir = os.path.join("data", "processed")
        if not os.path.exists(initial_dir): initial_dir = os.getcwd()
        
        path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("JSON", "*.json")])
        if path:
            try:
                with open(path, 'r', encoding="utf-8") as f:
                    self.resume_data = json.load(f)
                self.current_index = 0
                self.update_display()
                self.lbl_status.config(text="Veri Yüklendi. Analiz Bekleniyor.", fg="orange")
                # Skorları temizle
                self.lbl_hybrid_score.config(text="-%", fg="black")
                self.lbl_pure_score.config(text="-%", fg="black")
                self.lbl_hybrid_total.config(text="-%", fg="black")
                self.lbl_pure_total.config(text="-%", fg="black")
                
                # İstatistikleri Sıfırla
                self.total_processed = 0
                self.total_hybrid_acc = 0.0
                self.total_pure_acc = 0.0
            except Exception as e:
                messagebox.showerror("Hata", str(e))

    def update_display(self):
        if not self.resume_data: return
        self.text_area.delete("1.0", tk.END)
        current_content = self.resume_data[self.current_index].get('content', '')
        self.text_area.insert(tk.END, current_content)
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_counter.config(text=f"{self.current_index + 1} / {len(self.resume_data)}")

    def calculate_accuracy(self, doc, ground_truth_set):
        """Verilen doc objesi ile ground truth arasındaki başarıyı ölçer."""
        hits = 0
        total_expected = len(ground_truth_set)
        
        if total_expected == 0: return 100.0 if len(doc.ents) == 0 else 0.0

        for ent in doc.ents:
            # Basit eşleşme kontrolü (Label, Text)
            if (ent.label_, ent.text.strip()) in ground_truth_set:
                hits += 1
        
        return (hits / total_expected) * 100

    def process_current_resume(self):
        if not self.nlp or not self.resume_data: return

        text = self.text_area.get("1.0", tk.END).strip()
        
        # --- 1. GROUND TRUTH HAZIRLA ---
        current_annotations = self.resume_data[self.current_index].get("annotation", [])
        ground_truth_set = set()
        for ann in current_annotations:
            lbl = ann["label"][0]
            points = ann["points"][0]
            truth_text = points.get("text", text[points["start"]:points["end"]+1]).strip()
            ground_truth_set.add((lbl, truth_text))

        # --- 2. HİBRİT ANALİZ (TÜM PIPE'LAR AKTİF) ---
        doc_hybrid = self.nlp(text)
        acc_hybrid = self.calculate_accuracy(doc_hybrid, ground_truth_set)

        # --- 3. SAF ML ANALİZİ (RULE KAPALI) ---
        # Spacy'de bir pipe'ı geçici olarak kapatmak için disable_pipes kullanılır
        # "entity_ruler" bizim eklediğimiz kural katmanının adı
        try:
            with self.nlp.disable_pipes("entity_ruler"):
                doc_pure = self.nlp(text)
                acc_pure = self.calculate_accuracy(doc_pure, ground_truth_set)
        except:
            # Eğer pipeline isminde sorun varsa fallback yap
            # (entity_ruler yoksa zaten pure ML çalışır)
            doc_pure = self.nlp(text) 
            acc_pure = acc_hybrid # Ruler yoksa eşittir

        # --- 4. GÖRSELLEŞTİRME (HİBRİT SONUÇLARI GÖSTER) ---
        
        # Highlight ve Tablo
        for i in self.tree.get_children(): self.tree.delete(i)
        
        # Önce highlightları temizle (tag_remove tüm metin için zor olabilir, yeniden çizmek daha kolay)
        # Basitlik için sadece üstüne yazıyoruz
        
        for ent in doc_hybrid.ents:
            # Highlight Logic
            start_idx = "1.0"
            while True:
                start_idx = self.text_area.search(ent.text, start_idx, stopindex=tk.END)
                if not start_idx: break
                end_idx = f"{start_idx}+{len(ent.text)}c"
                self.text_area.tag_add(ent.label_, start_idx, end_idx)
                start_idx = end_idx
            
            # Tabloya Ekle
            is_correct = (ent.label_, ent.text.strip()) in ground_truth_set
            status_icon = "✅" if is_correct else "⚠️"
            self.tree.insert("", tk.END, values=(ent.text, ent.label_, status_icon))

        # --- 5. SKORLARI GÜNCELLE ---
        self.total_processed += 1
        self.total_hybrid_acc += acc_hybrid
        self.total_pure_acc += acc_pure
        
        avg_hybrid = self.total_hybrid_acc / self.total_processed
        avg_pure = self.total_pure_acc / self.total_processed

        self.lbl_hybrid_score.config(text=f"%{acc_hybrid:.1f}", fg="#2E7D32") # Yeşil
        self.lbl_hybrid_total.config(text=f"%{avg_hybrid:.1f}", fg="#2E7D32")
        
        # Renk mantığı: Pure ML hibrit'ten düşükse kırmızı yap (kötü olduğunu vurgula)
        pure_color = "#D32F2F" if acc_pure < acc_hybrid else "gray"
        self.lbl_pure_score.config(text=f"%{acc_pure:.1f}", fg=pure_color)
        
        pure_total_color = "#D32F2F" if avg_pure < avg_hybrid else "gray"
        self.lbl_pure_total.config(text=f"%{avg_pure:.1f}", fg=pure_total_color)

        self.lbl_status.config(text=f"Analiz Tamamlandı (Toplam: {self.total_processed})", fg="blue")

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