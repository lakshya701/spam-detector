"""
Export the trained Linear SVM + TF-IDF vocabulary to docs/model.json
so the browser demo (docs/index.html) can classify messages client-side.
Also writes docs/test_cases.json to check the browser matches Python.
"""

import json
import os
import re
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

STOPWORDS = set("""
a an the is are was were be been being to of in on for and or but if with
at by from up down out about into over after before this that these those
it its i you he she we they them his her our your my me us as not no do
does did doing have has had having will would can could should shall
""".split())


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(t for t in text.split() if t not in STOPWORDS)


df = pd.read_csv("data/sms_spam.csv").drop_duplicates(subset="message")
df["clean_message"] = df["message"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X_train_text, X_test_text, y_train, y_test, _, msg_test = train_test_split(
    df["clean_message"], df["label_num"], df["message"],
    test_size=0.2, random_state=42, stratify=df["label_num"],
)

# Simple ASCII token pattern so the browser tokenizes identically.
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True,
                             token_pattern=r"[A-Za-z0-9_]{2,}")
X_train = vectorizer.fit_transform(X_train_text)
model = LinearSVC(class_weight="balanced").fit(X_train, y_train)

vocab = vectorizer.vocabulary_
idx = sorted(vocab, key=vocab.get)
coef = model.coef_[0]
idf = vectorizer.idf_

os.makedirs("docs", exist_ok=True)
with open("docs/model.json", "w", encoding="utf-8") as f:
    json.dump({
        "stopwords": sorted(STOPWORDS),
        "punctuation": string.punctuation,
        "intercept": round(float(model.intercept_[0]), 6),
        "terms": {t: [round(float(idf[vocab[t]]), 5), round(float(coef[vocab[t]]), 5)] for t in idx},
    }, f, separators=(",", ":"))

X_test = vectorizer.transform(X_test_text)
scores = model.decision_function(X_test)
acc = ((scores > 0).astype(int) == y_test.values).mean()
print(f"Exported {len(idx)} terms; test accuracy with export tokenizer: {acc:.4f}")

with open("docs/test_cases.json", "w", encoding="utf-8") as f:
    json.dump([{"message": m, "score": round(float(s), 4)} for m, s in zip(msg_test, scores)], f)
