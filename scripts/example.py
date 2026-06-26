#!/usr/bin/env python3

"""
Library usage example — sms_analysis package.

Run from any directory after `pip install -e .`:
    python scripts/example.py
"""

from sklearn.model_selection import train_test_split

from pathlib import Path

from sms_analysis import load_dataset
from sms_analysis.classification import SpamClassifier, evaluate

# ── 1. Load and deduplicate ───────────────────────────────────────────────────
dataset = load_dataset(Path(__file__).parent.parent / "assignment/3D8026C4.zip/SMSSpamCollection").deduplicated
print(f"Dataset: {len(dataset)} unique messages ({len(dataset.spam)} spam / {len(dataset.ham)} ham)")

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    dataset.texts, dataset.labels, test_size=0.2, stratify=dataset.labels, random_state=42
)

# ── 3. Fit classifier ─────────────────────────────────────────────────────────
clf = SpamClassifier("logistic_regression").fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
result = evaluate(y_test, clf.predict(X_test), clf.predict_proba(X_test))
print(result.summary())

# ── 5. Predict new messages ───────────────────────────────────────────────────
samples = [
    "Hey, are you coming to the party tonight?",
    "Hey, are you coming to the cinema tonight?",
    "I'll be home around 7, can you start dinner?",
    "Hi Toni! Kommst du heute zum Karaoke?",
    "WINNER! Claim your free prize now — call 08001234567!",
    "WINNER! You have been selected for a FREE cash prize. Call 0800 now!",
    "Sie haben GEWONNEN! Sie wurden für einen KOSTENLOSEN Geldpreis ausgewählt. Rufen Sie jetzt unter 0800 an!",
    "Can you pick up some milk on the way home?",
    "Kannst du vom Billa eine Butter mitbringen?",
    "Urgent! Can you pick up some butter from carrefour?",
    "Dringend! Kannst du vom Billa eine Butter mitbringen?",
    "FREE entry to win £1000 cash, text WIN to 80085",
    "FREE entry to win $1000 cash, text WIN to 09001234",
    "FREE entry to win 1000 £ cash, text WIN to 80085",
    "FREE entry to win 1000 EUR cash, text WIN to 09001234",
    "Claim your €500 reward at www.win2day.at immediately!",
    "Holen Sie sich jetzt Ihre 500-Euro-Prämie unter www.win2day.at!",
]
predictions = clf.predict(samples)
probas = clf.predict_proba(samples)

print("\nPredictions:")
for text, label, proba in zip(samples, predictions, probas):
    p_spam = proba[1]
    print(f"  [{label.upper():4s}  p={p_spam:.2f}]  {text[:60]}")

# ── 6. Top features (what drives the model) ───────────────────────────────────
features = clf.top_features(n=10)
print("\nTop SPAM indicators:", [w for w, _ in features["spam"]])
print("Top HAM  indicators:", [w for w, _ in features["ham"]])
