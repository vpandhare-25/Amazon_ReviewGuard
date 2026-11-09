"""Step 3: tune hyperparameters and study feature importance.

1. Feature importance of the baseline model (saved as a CSV and a chart).
2. Grid search: 768 hyperparameter combinations x 5-fold cross-validation on the
   training set (3,840 fits), scored by ROC-AUC. The test set is never used to choose.
3. The best model is scored once on the held-out test set and compared to the baseline.
4. Feature selection check: retrain with only the features above an importance
   threshold and see whether the test score holds.

Run train_model.py first. Writes model/tuned_classifier.pkl, model/tuning_results.json,
model/feature_importance.csv and model/feature_importance.png.
Full grid search takes a while (about 15-30 minutes on a laptop); use --quick for a
small grid that checks the script runs.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_selection import SelectFromModel  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from train_model import evaluate, load_features, split  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

FULL_GRID = {
    "n_estimators": [50, 100, 200, 500],
    "learning_rate": [0.1, 0.3, 0.6, 1.0],
    "max_depth": [3, 6, 10],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.1, 0.5, 1.0, 1.5],
}  # 4 * 4 * 3 * 4 * 4 = 768 combinations
QUICK_GRID = {"n_estimators": [100], "learning_rate": [0.1, 0.3], "max_depth": [3, 4]}


def save_feature_importance(model, columns):
    importance = pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ordered = importance.iloc[::-1]  # biggest at the top of a horizontal bar chart
    ax.barh(ordered["feature"], ordered["importance"], color="#2a78d6")
    ax.set_xlabel("XGBoost feature importance (gain share)")
    ax.set_title("Which features the baseline model relies on")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    return importance


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    grid = QUICK_GRID if quick else FULL_GRID

    X, y = load_features()
    X_train, X_test, y_train, y_test = split(X, y)
    baseline = joblib.load(MODEL_DIR / "review_classifier.pkl")
    baseline_metrics = evaluate(baseline, X_test, y_test)

    # ---- 1. Feature importance ----
    importance = save_feature_importance(baseline, X.columns)
    print("Feature importance (baseline):")
    print(importance.to_string(index=False))

    # ---- 2. Grid search on the training set only ----
    search = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid=grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    start = time.time()
    search.fit(X_train, y_train)
    minutes = (time.time() - start) / 60
    n_combinations = len(search.cv_results_["params"])
    print(f"\nGrid search: {n_combinations} combinations x 5 folds in {minutes:.1f} min")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {search.best_score_:.4f}")

    # ---- 3. Score the best model once on the test set ----
    tuned = search.best_estimator_
    tuned_metrics = evaluate(tuned, X_test, y_test)
    print("\nTest set, baseline vs tuned:")
    print(f"  baseline  accuracy {baseline_metrics['accuracy']:.4f}  ROC-AUC {baseline_metrics['roc_auc']:.4f}")
    print(f"  tuned     accuracy {tuned_metrics['accuracy']:.4f}  ROC-AUC {tuned_metrics['roc_auc']:.4f}")
    print(tuned_metrics["report"])

    # ---- 4. Feature selection: keep features with importance >= 0.02 ----
    selector = SelectFromModel(baseline, threshold=0.02, prefit=True)
    kept = X.columns[selector.get_support()].tolist()
    selected = XGBClassifier(**search.best_params_, eval_metric="logloss", random_state=42)
    selected.fit(X_train[kept], y_train)
    selected_metrics = evaluate(selected, X_test[kept], y_test)
    print(f"\nWith only the {len(kept)} features above 0.02 importance: "
          f"accuracy {selected_metrics['accuracy']:.4f}  ROC-AUC {selected_metrics['roc_auc']:.4f}")

    if not quick:
        joblib.dump(tuned, MODEL_DIR / "tuned_classifier.pkl")
    results = {
        "grid": "quick" if quick else "full",
        "combinations": n_combinations,
        "cv_folds": 5,
        "search_minutes": round(minutes, 1),
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "baseline_test": {k: baseline_metrics[k] for k in ("accuracy", "roc_auc")},
        "tuned_test": {k: tuned_metrics[k] for k in ("accuracy", "roc_auc", "confusion_matrix")},
        "feature_selection": {
            "threshold": 0.02,
            "features_kept": kept,
            "test": {k: selected_metrics[k] for k in ("accuracy", "roc_auc")},
        },
    }
    out = MODEL_DIR / ("tuning_results_quick.json" if quick else "tuning_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")
