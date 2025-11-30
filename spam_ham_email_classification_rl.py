# Spam Ham Email Classification
## Name: Ritik Sanjay Patel
"""
Downloading Libraries:
- conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
- conda install matplotlib numpy -y
- pip install seaborn tqdm tensorflow
- pip install scikit-learn

Downloading Dataset:
- Dataset file is already avaiable in the local floder so theres no need to download it


"""

# Importing Libraries
import os
import pandas as pd
import numpy as np
import re, string
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plt
import seaborn as sns

import six
from io import StringIO

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ---------------------------
# Setup output folder
# ---------------------------
os.makedirs('outputs', exist_ok=True)

"""# Reading And Understanding Data"""

df = pd.read_csv("spam_Emails_data.csv")

df.head()

df.shape

df.info()

df.describe()

df.columns.tolist()

df.isnull().sum()

"""So there are two rows in my data which have null valeus in text column.

# Data Cleaning And Performing EDA
"""

# droping the rows with missing values
df = df.dropna(subset=["text"]).reset_index(drop=True)

"""In my case I have only two row which have null values. So I am dropping them."""

# creating a user define function named clean_text
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'<.*?>', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\d+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

"""I have created a user define function name clean_text to remove the unneccessary things from the text like urls, html tags, numbers, extra spaces and so on. I have taken this step in order to improve the quality of data."""

# Applying cleaning
df["clean"] = df["text"].astype(str).apply(clean_text)

df["label"] = df["label"].map({"Spam": 1, "Ham": 0})

"""Here I have converted the lables into a binary representation that spam is represented using 1 and ham is represented using 0"""

df.head()

print("\nLabel distribution:\n", df["label"].value_counts())

# Label distribution
df["label"].value_counts().plot(kind="bar", color=["skyblue", "lightcoral"])
plt.title("Spam vs Ham Distribution")
plt.xticks(ticks=[0,1], labels=["Ham (0)", "Spam (1)"], rotation=0)
plt.ylabel("Count")
filename = f"img1_lable_distribution.png"
plt.savefig(os.path.join("outputs", filename), bbox_inches="tight")
plt.close()

"""There are around 102k ham emails and 92k sapm emails.

This shows that the dataset is fairly balanced, meaning my model won't suffer from class imbalace and it makes evaluation metrics more reliable.
"""

# Email length analysis
df["length"] = df["clean"].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="length", hue="label", bins=50, kde=True)
plt.title("Email Length Distribution (Spam vs Ham)")
plt.xlabel("Email Length (characters)")
filename = f"img2_length_analysis.png"
plt.savefig(os.path.join("outputs", filename), bbox_inches="tight")
plt.close()

"""From the above histogram I can see that most of the ham and spam emails have short lengths, clustered under few thousands.

However there are some extreme outlires in ham which is making x-axis streach and compress around the main region.
"""

print(df.groupby("label")["length"].describe())

df = df[df["length"] < 5000].reset_index(drop=True)

"""Here I have handled the outliers present in the dataset as removing outlires will improve model stability and vizualization clarity."""

# Email length analysis
df["length"] = df["clean"].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="length", hue="label", bins=50, kde=True)
plt.title("Email Length Distribution (Spam vs Ham)")
plt.xlabel("Email Length (characters)")
filename = f"img3_new_length_analysis.png"
plt.savefig(os.path.join("outputs", filename), bbox_inches="tight")
plt.close()

"""Now dataset looks clean, balanced, and free from extreme outliers"""

print("\nLabel distribution:\n", df["label"].value_counts())

"""After cleaning and removing outliers, the dataset contains about 182k emails, almost evenly split between Ham and Spam.

# Splitting into Train and Test set
"""

# splitting data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean"], df["label"].astype(int),
    test_size=0.2, random_state=42, stratify=df["label"]
)

"""I have split the data into train and test split using sklearn train_test_split. I am going to use exact same train test split for all models for fair commparision across all the models.

# TF-IDF for Logistic Regression and Desicion Tree
"""

len(X_train_text), len(X_test_text)

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf  = tfidf.transform(X_test_text)

"""I am going to use same TF-IDF features for logistic regression model and descicision tree model. But for Glove I will use tokenizer along with padded sequence and load pre-trained Glove into embedding layer.

# Building and Evaluating Logistic Regression Model
"""

# Createing logistic regression model
lr_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

"""Here I have used max_iter=2000 which increases itretation to ensure convergence of large data. I have also used class_weight="balanced" to slightly adjust the weights if Ham > Spam or vice-versa"""

# Training the model
lr_model.fit(X_train_tfidf, y_train)

# making predection on test data
lr_pred = lr_model.predict(X_test_tfidf)

"""This gives an array on 0 and 1. 0 means ham and 1 means spam"""

def report_results(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"{name} -> Accuracy: {acc:.4f} | Precision: {pr:.4f} | Recall: {rc:.4f} | F1-Score: {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=["Ham (0)", "Spam (1)"]))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"])
    plt.title(f"{name} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    filename = f"{name}_Confusion Matrix.png"
    plt.savefig(os.path.join("outputs", filename), bbox_inches="tight")
    plt.close()

"""Here I have created a user define function report_resluts to get Accuraccy, Presicion, recall, F1-Score and confusion matrix. I have created a user define function so that I can reuse it for all the models."""

# Evaluating the Performance of the model
report_results("Logistic Regression (TF-IDF)", y_test, lr_pred)

"""#### Interpretation of the Model Performance
- Accuracy: The overall accuracy is around 97.99%. This implies that the logistics regressions desicision boundary between spam and ham is very well learned from the TF-IDF features.
- Precision: When the models predicts spam its correct 97.24% of times. This is important in email filtering as users hate losing legitimate emails to the spam folder.
- Recall: The model catches 98.59 % of all actual spam mails. This makes the filter highly reliable in catching almost all spam messages.
- F1-Score: 97.91%

#### Conclusion for Logistic Regression

The Logistic Regression model with TF-IDF features provides highly accurate, balanced, and reliable email classification.
It correctly identifies spam while maintaining very low false positive and false negative rates.

# Building and Evaluating Desicision Tree Classifire Model
"""

# Creating Decision Tree model
dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    max_depth=40,
    min_samples_leaf=2
)

"""- Here I have used class_weight="balanced" to prevent slight imbalance
- I have set max_depth to 40 to prevent overfitting
- I have also set min_samples_leaf=2 to ensure that each leaf has atlest 2 samples
"""

# training the model
dt_model.fit(X_train_tfidf, y_train)

# making desicisions on test data
dt_pred = dt_model.predict(X_test_tfidf)

# Evaluating the performance of the model
report_results("Decision Tree (TF-IDF)", y_test, dt_pred)

"""#### Interpretation of the Model Performance
- Accuracy: The overall accuracy is around 94.63%. This is slightly lower than Logistic Regression (97.99%), which is expected as trees can overfit TF-IDF data.
- Precision: When the models predicts spam its correct 91.74% of times. This implies that the tree makes more false spam predictions
- Recall: The model catches 97.54 % of all actual spam mails. This makes the filter highly reliable in catching almost all spam messages.
- F1-Score: 94.55%

#### Conclusion for desicsion tree model
The Decision Tree model demonstrates strong spam detection capability with 94.6% accuracy and high recall.

# Building and Evaluating Glove Model
"""

# Tokenizing and Padding
MAX_VOCAB = 50000     # cap vocabulary size
MAX_LEN   = 200       # sequence length

tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tok.fit_on_texts(X_train_text)

X_train_seq = tok.texts_to_sequences(X_train_text)
X_test_seq  = tok.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding="post", truncating="post")

y_train_dl = y_train.values if hasattr(y_train, "values") else y_train
y_test_dl  = y_test.values  if hasattr(y_test, "values")  else y_test

len(tok.word_index), X_train_pad.shape, X_test_pad.shape

# Loading Glove and building the embedding matrix
EMB_DIM = 100
GLOVE_PATH = "glove.6B.100d.txt"

emb_index = {}
with open(GLOVE_PATH, encoding="utf8") as f:
    for line in f:
        values = line.rstrip().split(" ")
        word = values[0]
        vec  = np.asarray(values[1:], dtype="float32")
        emb_index[word] = vec

word_index = tok.word_index
vocab_size = min(MAX_VOCAB, len(word_index) + 1)

emb_matrix = np.random.normal(scale=0.6, size=(vocab_size, EMB_DIM)).astype(np.float32)
for w, i in word_index.items():
    if i < vocab_size and w in emb_index:
        emb_matrix[i] = emb_index[w]

vocab_size

# Defining the model (Embedding + LSTM)
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=EMB_DIM,
        weights=[emb_matrix],
        input_length=MAX_LEN,
        trainable=False
    ),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Training with early stopping
es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train_dl,
    validation_split=0.1,
    epochs=6,
    batch_size=256,
    callbacks=[es],
    verbose=1
)

# Evaluating the performance of the model
dl_probs = model.predict(X_test_pad).ravel()
dl_pred  = (dl_probs >= 0.5).astype(int)

report_results("Deep Learning (GloVe + LSTM)", y_test_dl, dl_pred)

"""#### Interpretation of the Model Performance
- Accuracy: The overall accuracy is around 93.80%. This is similar to Logistic Regression (97.99%) and Desicision Tree (94.6).
- Precision: When the models predicts spam its correct 92.92% of times.
- Recall: The model catches 94.2 % of all actual spam mails. This makes the filter highly reliable in catching almost all spam messages.
- F1-Score: 93.56%

# Comparing Logistic Regression, Desicision Tree and Glove Model
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def row(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"Model": name, "Accuracy": acc, "Precision": pr, "Recall": rc, "F1": f1}

rows = []
rows.append(row("Logistic Regression (TF-IDF)", y_test, lr_pred))
rows.append(row("Decision Tree (TF-IDF)", y_test, dt_pred))
rows.append(row("DL (GloVe + LSTM)", y_test_dl, dl_pred))

pd.DataFrame(rows).sort_values("F1", ascending=False)

"""The results show that the Logistic Regression model achieved the highest performance, with an accuracy of 97.98% and an F1-score of 97.9%. The Decision Tree followed with 94.6% accuracy, while the Deep Learning (GloVe + LSTM) model reached 93.8% accuracy. These results indicate that for structured, keyword-based datasets such as spam detection, traditional models like Logistic Regression often outperform more complex neural architectures."""

