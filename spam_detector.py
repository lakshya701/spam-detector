"""
Spam Mail Detector
Classifies SMS messages as spam or ham (not spam).
Dataset: UCI SMS Spam Collection (5,574 real messages).
"""

import os
import re
import string
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/sms_spam.csv")
df = df.drop_duplicates(subset="message")  # dataset has ~400 repeated messages

print("Dataset shape:", df.shape)
print(df["label"].value_counts(), "\n")

# ---------------------------------------------------------------
# 2. Preprocess text
# ---------------------------------------------------------------
STOPWORDS = set("""
a an the is are was were be been being to of in on for and or but if with
at by from up down out about into over after before this that these those
it its i you he she we they them his her our your my me us as not no do
does did doing have has had having will would can could should shall
""".split())


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # strip urls
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


df["clean_message"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print("Example before/after cleaning:")
print(" raw   :", df["message"].iloc[0])
print(" clean :", df["clean_message"].iloc[0], "\n")

# ---------------------------------------------------------------
# 3. Feature extraction — TF-IDF
# ---------------------------------------------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------------------------------------------------------
# 4. Train and compare models
# ---------------------------------------------------------------
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
}

results = []
best_name, best_f1, best_model, best_preds = None, -1, None, None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results.append({"model": name, "accuracy": round(acc, 4), "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4)})
    if f1 > best_f1:
        best_name, best_f1, best_model, best_preds = name, f1, model, preds

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print("Model comparison:\n", results_df, "\n")
results_df.to_csv("outputs/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 5. Evaluate best model in detail
# ---------------------------------------------------------------
print(f"Best model: {best_name} (F1={best_f1:.4f})\n")
print("Classification report:\n", classification_report(y_test, best_preds, target_names=["ham", "spam"]))

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=140)
plt.close()

# ---------------------------------------------------------------
# 6. Try it on a few new messages
# ---------------------------------------------------------------
sample_messages = [
    "Congratulations! You have won a free iPhone, call now to claim your prize!",
    "Hey, are you free for lunch tomorrow around noon?",
    "URGENT: verify your account now or it will be suspended today.",
    "Can you send me the notes from yesterday's meeting?",
]

sample_clean = [clean_text(m) for m in sample_messages]
sample_vec = vectorizer.transform(sample_clean)
sample_preds = best_model.predict(sample_vec)

print("\nSample predictions:")
for msg, pred in zip(sample_messages, sample_preds):
    label = "SPAM" if pred == 1 else "ham"
    print(f"  [{label}] {msg}")

print("\nSaved: outputs/model_comparison.csv, outputs/confusion_matrix.png")
