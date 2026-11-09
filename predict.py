"""Scores a single review with the saved model.

Usage:
    python predict.py "This product is great! I love it." --rating 5 --category Electronics_5

Prints the probability that the review was written by a human (OR) vs generated (CG).
Uses model/tuned_classifier.pkl if it exists, otherwise model/review_classifier.pkl.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from feature_extraction import extract_features  # noqa: E402
from preprocess import preprocess_text  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("text")
parser.add_argument("--rating", type=float, required=True, help="star rating, 1-5")
parser.add_argument("--category", required=True, help="e.g. Books_5, Electronics_5, Home_and_Kitchen_5")
args = parser.parse_args()

model_dir = ROOT / "model"
model_path = model_dir / "tuned_classifier.pkl"
if not model_path.exists():
    model_path = model_dir / "review_classifier.pkl"
model = joblib.load(model_path)
feature_names = json.loads((model_dir / "feature_names.json").read_text())

df = pd.DataFrame({"text_": [args.text], "rating": [args.rating], "category": [args.category]})
df["cleaned_text"] = df["text_"].apply(preprocess_text)
df = extract_features(df, include_pos=True)
X = pd.get_dummies(df.drop(columns=["text_", "cleaned_text"]), columns=["category"])
X = X.reindex(columns=feature_names, fill_value=0)  # same columns, same order as training

prob_human = float(model.predict_proba(X)[0, 1])
label = "human-written (OR)" if prob_human >= 0.5 else "computer-generated (CG)"
print(f"{label}: P(human) = {prob_human:.3f}  [model: {model_path.name}]")
