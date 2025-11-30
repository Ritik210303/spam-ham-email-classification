# 📧 Spam vs Ham Email Classification  
### Logistic Regression (TF-IDF) | Decision Tree | GloVe + LSTM (Deep Learning)

This project builds a complete email classification system that detects **Spam (1)** vs **Ham (0)** using both **traditional machine learning** and **deep learning (LSTM + GloVe embeddings)**.  
The goal is to compare performance across models and identify the most effective approach for spam detection.

---

## 📂 Repository Structure

```
spam-ham-email-classification/
│
├── spam_ham_email_classification.ipynb      # Full notebook (EDA + ML + DL)
├── spam_ham_email_classification_rl.py      # Python script version
├── ReadMe.pdf                                # Academic project report
│
├── spam_Emails_data.csv                      # Dataset
├── glove.6B.100d.txt                         # Pre-trained GloVe embeddings (100d)
│
├── outputs/                                  # All charts & model results
│   ├── img1_lable_distribution.png
│   ├── img2_length_analysis.png
│   ├── img3_new_length_analysis.png
│   ├── Logistic Regression (TF-IDF)_Confusion Matrix.png
│   ├── Decision Tree (TF-IDF)_Confusion Matrix.png
│   ├── Deep Learning (GloVe + LSTM)_Confusion Matrix.png
│   └── comparison_table.png
│
└── README.md
```

---

## 🧠 Problem Statement

Given a dataset of real-world emails, the objective is to classify each message as **Spam** or **Ham** using:

- Logistic Regression (TF-IDF)
- Decision Tree (TF-IDF)
- Deep Learning (GloVe Embeddings + LSTM)

This project includes full preprocessing, training, evaluation, interpretation, and visualization.

---

## 📊 Dataset

- Source: Public Spam/Ham dataset (~190K emails)  
- Final processed dataset: **182K samples**  
- Columns:
  - `text` — email body  
  - `label` — `{0: ham, 1: spam}`  

---

## 🧹 Preprocessing Pipeline

✔ Lowercasing  
✔ Remove URLs & HTML tags  
✔ Remove punctuation & numbers  
✔ Remove stopwords  
✔ Tokenization  
✔ Email length analysis + outlier removal (>5000 chars)  
✔ Label encoding  
✔ Train–test split (80/20)  
✔ Shared across all models for **fair comparison**

---

## 🛠 Models Implemented

### 1️⃣ Logistic Regression (TF-IDF)
- TF-IDF with 50,000 features  
- N-grams: (1,2)  
- `max_iter = 2000`  
- `class_weight = "balanced"`  
- **Best-performing model**  

---

### 2️⃣ Decision Tree (TF-IDF)
- `max_depth = 40`  
- `min_samples_leaf = 2`  
- Good recall but overfits slightly  

---

### 3️⃣ Deep Learning — GloVe + LSTM
- Pretrained **GloVe.6B.100d** embeddings  
- Embedding Matrix built from vocabulary  
- 64-unit LSTM  
- Dropout 0.2  
- Batch size 256  
- Epochs 6 with EarlyStopping  
- Suitable for semantic text understanding  

---

## 📈 Model Evaluation

| Model                     | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| **Logistic Regression**  | **97.98%** | 97.24% | 98.59% | **97.91%** |
| Decision Tree            | 94.63% | 91.74% | 97.54% | 94.55% |
| GloVe + LSTM             | 93.80% | 92.92% | 94.19% | 93.56% |

**Conclusion:**  
➡️ Logistic Regression (TF-IDF) performs the best because spam detection relies heavily on **keyword patterns**, which TF-IDF captures very well.  
➡️ LSTM performs well but cannot outperform TF-IDF on short, keyword-heavy emails.

---

## 📊 Visual Outputs (in `/outputs` folder)

- Label distribution  
- Email length distribution  
- TF-IDF confusion matrices  
- LSTM confusion matrix  
- Model comparison table  
- Training curves for deep learning  

---

## ▶️ How to Run the Project

### **Option A — Run Notebook**
Open:

```
spam_ham_email_classification.ipynb
```

Make sure the following files are in the same directory:

- `spam_Emails_data.csv`
- `glove.6B.100d.txt`

---

### **Option B — Run Python Script**

Install dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn tensorflow
```

Run:

```bash
python spam_ham_email_classification_rl.py
```

Outputs will be stored in the **outputs/** folder.

---

## 🎓 Skills Demonstrated

- NLP Preprocessing  
- TF-IDF Feature Engineering  
- Logistic Regression & Decision Trees  
- Deep Learning with LSTM  
- GloVe Embeddings  
- Confusion Matrices  
- Performance Comparison  
- EDA & Visualization  

---

## 📄 License
This project is for academic and learning purposes.

