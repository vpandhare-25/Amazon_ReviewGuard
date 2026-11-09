"""Step 2: train the baseline XGBoost classifier and save it.

Task: tell computer-generated reviews (CG, label 0) from original human-written
reviews (OR, label 1), using the 18 features in processed-dataset.csv.

Writes:
  model/review_classifier.pkl   trained model (Joblib)
  model/feature_names.json      feature column order the model expects
  model/model_metadata.json     test metrics and dataset sizes
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
LABEL_MAP = {"OR": 1, "CG": 0}  # OR = original (human), CG = computer-generated


def load_features():
    """Returns X (features) and y (labels). Shared by train_model.py and tune_model.py."""
    df = pd.read_csv(ROOT / "processed-dataset.csv")
    # Drop the label and raw text; one-hot encode the product category into 10 columns.
    X = pd.get_dummies(df.drop(columns=["label", "text_", "cleaned_text"]), columns=["category"])
    y = df["label"].map(LABEL_MAP)
    return X, y


def split(X, y):
    """80/20 train/test split with a fixed seed so every script uses the same test set."""
    return train_test_split(X, y, random_state=42, test_size=0.2)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]  # probability the review is human-written
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # rows: true CG, true OR
        "report": classification_report(y_test, pred, target_names=["CG (AI)", "OR (human)"], digits=3),
    }


if __name__ == "__main__":
    X, y = load_features()
    print(f"{len(X)} reviews, {X.shape[1]} features, labels: {y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = split(X, y)

    # 100 trees, each at most 4 levels deep (shallow trees overfit less).
    model = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print(metrics["report"])
    print(f"Confusion matrix (rows = true CG, true OR): {metrics['confusion_matrix']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}   ROC-AUC: {metrics['roc_auc']:.4f}")

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "review_classifier.pkl")
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(X.columns.tolist(), f)
    metadata = {
        "test_accuracy": metrics["accuracy"],
        "test_auc_score": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "num_features": X.shape[1],
        "label_mapping": LABEL_MAP,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }
    with open(model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model saved in {model_dir}/")
