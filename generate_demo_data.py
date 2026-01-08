import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import spacy
import os

class ResumeNERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume NER Intelligence - Demo (Academic Version)")
        self.root.geometry("1200x750")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Veri Değişkenleri
        self.nlp = None
        self.resume_data = []
        self.current_index = 0
        
        # İstatistik Değişkenleri
        self.total_processed = 0
        self.total_score_acc = 0.0 # Toplam birikmiş skor
        
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

        btn_load_model = tk.Button(toolbar, text="📂 1. Modeli Yükle", command=self.load_model, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_load_model.pack(side=tk.LEFT, padx=5, pady=5)

        btn_load_json = tk.Button(toolbar, text="📄 2. Test Verisi Yükle", command=self.load_json, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        btn_load_json.pack(side=tk.LEFT, padx=5, pady=5)

        # SKOR PANELI (YENİ)
        self.score_frame = tk.Frame(toolbar, bg="#e0e0e0")
        self.score_frame.pack(side=tk.RIGHT, padx=20)
        
        self.lbl_current_acc = tk.Label(self.score_frame, text="Anlık Başarı: -", font=("Arial", 12, "bold"), fg="#333", bg="#e0e0e0")
        self.lbl_current_acc.pack(side=tk.LEFT, padx=10)
        
        self.lbl_total_acc = tk.Label(self.score_frame, text="Ortalama Başarı: -", font=("Arial", 12, "bold"), fg="#00008B", bg="#e0e0e0")
        self.lbl_total_acc.pack(side=tk.LEFT, padx=10)

        # --- ANA ALAN ---
        paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # SOL: Metin
        left_frame = ttk.LabelFrame(paned_window, text="Özgeçmiş İçeriği")
        paned_window.add(left_frame, weight=3) # Sol taraf biraz daha geniş
        
        self.text_area = tk.Text(left_frame, wrap=tk.WORD, font=("Consolas", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        for label, color in self.entity_colors.items():
            self.text_area.tag_config(label, background=color)

        # SAĞ: Tablo
        right_frame = ttk.LabelFrame(paned_window, text="Çıkarılan Varlıklar & Doğrulama")
        paned_window.add(right_frame, weight=2)

        columns = ("Entity Text", "Label", "Status")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings")
        self.tree.heading("Entity Text", text="Değer")
        self.tree.heading("Label", text="Etiket")
        self.tree.heading("Status", text="Durum")
        
        self.tree.column("Entity Text", width=150)
        self.tree.column("Label", width=120)
        self.tree.column("Status", width=80)
        
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
        
        btn_process = tk.Button(nav_frame, text="⚙️ ANALİZ ET & PUANLA", command=self.process_current_resume, bg="#FF5722", fg="white", font=("Arial", 11, "bold"))
        btn_process.pack(side=tk.RIGHT, padx=10)

    def load_model(self):
        initial_dir = "models"
        path = filedialog.askdirectory(initialdir=initial_dir, title="Model Seç")
        if path:
            self.nlp = spacy.load(path)
            messagebox.showinfo("Bilgi", f"Model Yüklendi: {os.path.basename(path)}")

    def load_json(self):
        initial_dir = os.path.join("data", "processed")
        if not os.path.exists(initial_dir): initial_dir = os.getcwd()
        path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("JSON", "*.json")])
        if path:
            with open(path, 'r', encoding="utf-8") as f:
                self.resume_data = json.load(f)
            self.current_index = 0
            self.total_processed = 0 # Yeni dosya yüklenince istatistikleri sıfırla
            self.total_score_acc = 0.0
            self.update_display()
            # Skorları sıfırla
            self.lbl_current_acc.config(text="Anlık Başarı: -")
            self.lbl_total_acc.config(text="Ortalama Başarı: -")

    def update_display(self):
        if not self.resume_data: return
        self.text_area.delete("1.0", tk.END)
        current_content = self.resume_data[self.current_index].get('content', '')
        self.text_area.insert(tk.END, current_content)
        for i in self.tree.get_children(): self.tree.delete(i)
        self.lbl_counter.config(text=f"{self.current_index + 1} / {len(self.resume_data)}")

    def process_current_resume(self):
        if not self.nlp or not self.resume_data: return

        text = self.text_area.get("1.0", tk.END).strip()
        doc = self.nlp(text)
        
        # --- DOĞRULAMA MANTIĞI (GROUND TRUTH VS PREDICTION) ---
        
        # 1. Gerçek Verileri (Ground Truth) Hazırla
        # Format: Set of (Label, Text) - Basit kıyaslama için
        current_annotations = self.resume_data[self.current_index].get("annotation", [])
        ground_truth_set = set()
        for ann in current_annotations:
            lbl = ann["label"][0]
            # Noktaları alıp metni çekelim (JSON'da text olmayabilir bazen)
            points = ann["points"][0]
            truth_text = points.get("text", text[points["start"]:points["end"]+1]).strip()
            ground_truth_set.add((lbl, truth_text))

        # 2. Model Tahminlerini Al
        predictions = []
        hits = 0
        
        for ent in doc.ents:
            # Highlight yap
            start_idx = "1.0"
            while True:
                start_idx = self.text_area.search(ent.text, start_idx, stopindex=tk.END)
                if not start_idx: break
                end_idx = f"{start_idx}+{len(ent.text)}c"
                self.text_area.tag_add(ent.label_, start_idx, end_idx)
                start_idx = end_idx
            
            # Doğruluk Kontrolü
            # Modelin bulduğu (Label, Text) gerçek setin içinde var mı?
            # Not: %100 string eşleşmesi arıyoruz. (Skills için split yapılmış haliyle)
            is_correct = (ent.label_, ent.text.strip()) in ground_truth_set
            
            # Bazen Skills'lerde kısmi eşleşme olabilir, demo için basit tutalım.
            # Eğer tam eşleşme yoksa ama text ground truth içinde geçiyorsa 'Yarım Puan' verilebilir ama
            # şimdilik katı kural (Exact Match) uygulayalım, çünkü akademik raporda da öyle dedik.
            
            status_icon = "✅" if is_correct else "⚠️"
            if is_correct: hits += 1
                
            predictions.append((ent.text, ent.label_, status_icon))

        # 3. Tabloyu Güncelle
        for i in self.tree.get_children(): self.tree.delete(i)
        for pred in predictions:
            self.tree.insert("", tk.END, values=pred)
            
        # 4. İstatistikleri Hesapla
        # Başarı = (Doğru Bulunanlar) / (Toplam Olması Gerekenler) -> Recall mantığı
        total_expected = len(ground_truth_set)
        
        # Sıfıra bölme hatası önlemi
        if total_expected > 0:
            current_accuracy = (hits / total_expected) * 100
        else:
            current_accuracy = 100 if hits == 0 else 0

        # Kümülatif (Ortalama) Hesaplama
        # Not: Aynı CV'ye tekrar basarsa ortalamayı bozmamak için basit bir logic:
        # Sadece ileri gittikçe ortalama güncellensin diyebiliriz ama demo için anlık hesaplayalım.
        self.total_processed += 1
        self.total_score_acc += current_accuracy
        average_accuracy = self.total_score_acc / self.total_processed

        # Ekrana Bas
        self.lbl_current_acc.config(text=f"Anlık Başarı: %{current_accuracy:.1f}")
        self.lbl_total_acc.config(text=f"Ortalama Başarı: %{average_accuracy:.1f}")

        # Renkli geri bildirim
        color = "green" if current_accuracy > 70 else "orange" if current_accuracy > 40 else "red"
        self.lbl_current_acc.config(fg=color)

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