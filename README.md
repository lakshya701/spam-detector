# Spam Mail Detector

Classifies SMS/email-style messages as **spam** or **ham** (not spam) using
text preprocessing + TF-IDF + a Naive Bayes / Logistic Regression classifier.

## Dataset
This sandbox has no internet access, so the real
[SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
couldn't be downloaded directly here. `make_dataset.py` instead generates
`data/sms_dataset.csv` — 240 template-based messages (120 spam / 120 ham) that
mirror the same real-world patterns: spam leans on urgency, prize claims,
"free", phone numbers and links; ham is ordinary scheduling/small-talk.

**To use the real dataset:** download `SMSSpamCollection` from the link above,
save it as `data/sms_dataset.csv` with columns `label,message`, and skip
`make_dataset.py`. Everything downstream works unchanged.

## Approach
1. **Load** the labeled messages.
2. **Preprocess** — lowercase, strip URLs and punctuation, remove a small
   stopword list, tokenize.
3. **Feature extraction** — TF-IDF with unigrams + bigrams (max 2000 features).
4. **Train & compare** two classifiers:
   - Multinomial Naive Bayes
   - Logistic Regression
5. **Evaluate** with accuracy, precision, recall, F1, and a confusion matrix
   (`outputs/confusion_matrix.png`).
6. **Sanity check** on four new, hand-written messages not in the training set.

## Results
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |

Scores are near-perfect because the generated dataset uses a fixed set of
templates, so vocabulary cleanly separates the two classes — this is expected
for template data and mainly demonstrates that the pipeline works end-to-end.
On the real SMS Spam Collection (noisier, more varied language) expect
accuracy in the 96–98% range instead, which is the honest number to report if
you swap in the real dataset.

## Run it yourself
```bash
pip install -r requirements.txt
python make_dataset.py      # builds data/sms_dataset.csv
python spam_detector.py     # trains, evaluates, saves outputs/
```

## Skills demonstrated
Text preprocessing, TF-IDF feature extraction, basic NLP, classification
modeling, and evaluation with precision/recall/F1 (more informative than
accuracy alone for spam detection, where false positives — real messages
marked as spam — matter more than raw accuracy suggests).
