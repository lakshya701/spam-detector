// Browser port of spam_detector.py's pipeline: clean -> TF-IDF (1-2 grams,
// sublinear tf, l2 norm) -> Linear SVM decision function.
// Weights come from model.json, written by export_model.py.

function createClassifier(model) {
  const stop = new Set(model.stopwords);
  const punct = new Set(model.punctuation);

  function clean(text) {
    text = text.toLowerCase().replace(/http\S+|www\.\S+/g, " ");
    text = [...text].filter((c) => !punct.has(c)).join("");
    return text.split(/\s+/).filter((t) => t && !stop.has(t)).join(" ");
  }

  function features(text) {
    const tokens = clean(text).match(/[A-Za-z0-9_]{2,}/g) || [];
    const grams = [...tokens];
    for (let i = 0; i + 1 < tokens.length; i++) grams.push(tokens[i] + " " + tokens[i + 1]);

    const counts = new Map();
    for (const g of grams) if (model.terms[g]) counts.set(g, (counts.get(g) || 0) + 1);

    const weights = new Map();
    let norm = 0;
    for (const [g, n] of counts) {
      const w = (1 + Math.log(n)) * model.terms[g][0];
      weights.set(g, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [g, w] of weights) weights.set(g, w / norm);
    return weights;
  }

  // Returns the SVM score (> 0 means spam) and each term's contribution.
  function classify(text) {
    let score = model.intercept;
    const contributions = [];
    for (const [g, w] of features(text)) {
      const c = w * model.terms[g][1];
      score += c;
      contributions.push({ term: g, value: c });
    }
    contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return { score, isSpam: score > 0, contributions };
  }

  return { classify };
}

if (typeof module !== "undefined") module.exports = { createClassifier };
