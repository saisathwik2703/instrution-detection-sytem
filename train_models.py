"""
train_models.py
----------------
Trains the two classification models the project brief calls for:
  1. Decision Tree
  2. Random Forest

Both are trained on the NSL-KDD training set (KDDTrain+) and evaluated on
the SEPARATE, held-out NSL-KDD test set (KDDTest+) — this is the standard
NSL-KDD benchmark split, and it is a much fairer measure of real-world
performance than a random train/test split of the same file, since the
test set deliberately contains attack patterns not seen during training.

Run with:
    python train_models.py
"""

import os
import time
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from constants import RANDOM_STATE
from preprocessing import run_preprocessing

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "..", "data")
MODEL_DIR = os.path.join(THIS_DIR, "..", "models")


def evaluate(name, model, X_test, y_test):
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Inference time for {len(y_test)} samples: {inference_time:.3f}s")
    print("\nConfusion matrix (rows=actual, cols=predicted) [normal, attack]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["normal", "attack"]))

    return {
        "name": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "confusion_matrix": cm.tolist(),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "KDDTrain.txt")
    test_path = os.path.join(DATA_DIR, "KDDTest.txt")

    print("Loading and preprocessing NSL-KDD data...")
    X_train, y_train, X_test, y_test, encoders, scaler, feature_columns = \
        run_preprocessing(train_path, test_path)

    results = []

    # ---------------- Decision Tree ----------------
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    results.append(evaluate("Decision Tree", dt, X_test, y_test))
    joblib.dump(dt, os.path.join(MODEL_DIR, "decision_tree.joblib"))

    # ---------------- Random Forest ----------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))

    # ---------------- Feature importance (Random Forest) ----------------
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 most important features (Random Forest):")
    for i in top_idx:
        print(f"  {feature_columns[i]:30s} {importances[i]:.4f}")

    # ---------------- Save a summary ----------------
    summary_path = os.path.join(MODEL_DIR, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("NSL-KDD Intrusion Detection — Training Summary\n")
        f.write("=" * 55 + "\n\n")
        for r in results:
            f.write(f"{r['name']}\n")
            f.write(f"  Accuracy : {r['accuracy']:.4f}\n")
            f.write(f"  Precision: {r['precision']:.4f}\n")
            f.write(f"  Recall   : {r['recall']:.4f}\n")
            f.write(f"  F1-score : {r['f1']:.4f}\n\n")

    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Summary written to: {summary_path}")
    print("\nDone. You can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    main()
