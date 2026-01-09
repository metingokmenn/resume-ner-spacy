# Information Extraction from Unstructured Resumes using NER

## 📌 Project Abstract

This project is a Natural Language Processing (NLP) study aiming to extract structured information (Name, Skills, School, Experience, etc.) from unstructured resume texts. Within the scope of the project, a custom Named Entity Recognition (NER) model was trained and tested using the Spacy library.

## 🛠️ Methodology

The project follows the academic pipeline below:

1.  **Data Collection:** 220 labeled resume entries sourced from Kaggle were used.
2.  **Preprocessing:**
    - Data cleaning (Whitespace and character corrections).
    - A special `Span Trimming` algorithm was developed for Alignment issues.
3.  **Data Splitting:** The dataset was randomly split into **80% Training (Train)** and **20% Test** to measure the model's generalization ability.
4.  **Model Training:**
    - **Architecture:** Transition-based NER (Spacy).
    - **Optimization:** Overfitting was prevented using `Compounding Batch Size` and `Dropout Decay` techniques.
5.  **Evaluation:** Precision, Recall, and F1-Score metrics were calculated on the test set.

## 📂 Project Structure

- `data/`: Raw datasets.
- `src/`: Source codes (Loader, Trainer, Evaluator).
- `models/`: Trained model outputs.
- `results/`: Performance charts and metric tables.

## 📊 Experimental Results

The success of the model on the test dataset is detailed in the `results/evaluation_metrics.csv` file. The general F1 score and label-based success distribution are presented in the `results/f1_score_chart.png` graph.

## 🚀 Installation and Usage

1. **Install Requirements:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Project:**

   ```bash
   python main.py
   ```

   This command automatically runs the data processing, training, and testing processes in order.

---

## Developer

**Name Surname:** Metin Gökmen

### 🚫 `.gitignore`

To keep the git repo clean:

```text
# Python virtual environment
venv/
__pycache__/
*.pyc

# Model files (Can be large)
models/

# Results (Reproducible)
results/

# System files
.DS_Store
```
