"""
evaluate_baseline.py
--------------------
Loads saved model artifacts and evaluates the baseline classifier on the
held-out test set. Reports accuracy, per-class precision/recall/F1, and a
confusion matrix.

Usage
-----
    python -m src.training.evaluate_baseline
  or
    python src/training/evaluate_baseline.py
"""

import os
import sys

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Allow running the script directly from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.preprocess import load_dataset, prepare_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = "../../data/raw/introduction/interview_responses.csv"
MODEL_PATH = "../../models/baseline/model.pkl"
VECTORIZER_PATH = "../../models/baseline/vectorizer.pkl"

# Display order for consistent confusion matrix readout
LABEL_ORDER = ["Poor", "Average", "Good", "Excellent"]


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_artifacts():
    """
    Load the persisted model and vectorizer from disk.

    Returns
    -------
    tuple : (model, vectorizer)

    Raises
    ------
    FileNotFoundError
        If either artifact file is missing. Run train_baseline.py first.
    """
    for path in (MODEL_PATH, VECTORIZER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: '{path}'. "
                "Please run train_baseline.py before evaluating."
            )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"[load_artifacts] Loaded model      ← '{MODEL_PATH}'")
    print(f"[load_artifacts] Loaded vectorizer ← '{VECTORIZER_PATH}'\n")
    return model, vectorizer


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_confusion_matrix(
    cm: np.ndarray,
    labels: list,
) -> None:
    """
    Render the confusion matrix as a readable ASCII table.

    Parameters
    ----------
    cm : np.ndarray
        Square matrix from sklearn's confusion_matrix (shape: n_classes × n_classes).
    labels : list
        Ordered class labels corresponding to matrix rows/columns.
    """
    col_width = max(len(lbl) for lbl in labels) + 2
    header = "Predicted →".ljust(col_width) + "".join(
        lbl.rjust(col_width) for lbl in labels
    )
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)
    for i, row_label in enumerate(labels):
        row = row_label.ljust(col_width) + "".join(
            str(cm[i, j]).rjust(col_width) for j in range(len(labels))
        )
        print(row)
    print(separator)


# ---------------------------------------------------------------------------
# Main evaluation workflow
# ---------------------------------------------------------------------------

def evaluate() -> None:
    """
    End-to-end evaluation pipeline:
        1. Load saved model and vectorizer
        2. Load dataset and recreate the identical test split
        3. Vectorize the test set
        4. Generate predictions
        5. Print accuracy, classification report, and confusion matrix
    """
    print("=" * 55)
    print("  AI Interview Analyzer — Baseline Evaluation")
    print("=" * 55)

    # 1. Load artifacts
    model, vectorizer = load_artifacts()

    # 2. Load dataset and recreate the exact same test split
    #    (same random_state=42 + stratify ensures identical splits)
    df = load_dataset(DATASET_PATH)
    _, X_test, _, y_test = prepare_data(df)

    # 3. Vectorize test set using the *fitted* vocabulary (no re-fitting)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. Predict
    y_pred = model.predict(X_test_tfidf)

    # 5a. Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'─' * 55}")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"{'─' * 55}\n")

    # 5b. Classification Report (precision / recall / F1 per class)
    # Restrict label order to classes actually present in test labels
    present_labels = [lbl for lbl in LABEL_ORDER if lbl in y_test.unique()]
    report = classification_report(
        y_test,
        y_pred,
        labels=present_labels,
        zero_division=0,
    )
    print("Classification Report")
    print("─" * 55)
    print(report)

    # 5c. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    print("Confusion Matrix  (rows = Actual, cols = Predicted)")
    _print_confusion_matrix(cm, present_labels)

    print("\n[evaluate] ✓ Evaluation complete.")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()