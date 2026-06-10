"""
train_baseline.py
-----------------
Trains a TF-IDF + Logistic Regression baseline classifier for the
AI Interview Analyzer. Saves the fitted vectorizer and model to disk
using joblib for later evaluation or serving.

Usage
-----
    python -m src.training.train_baseline
  or
    python src/training/train_baseline.py
"""

import os
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Allow running the script directly from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.preprocess import load_dataset, prepare_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = "data/raw/interview_responses_batch_01.csv"
MODEL_DIR = "models/baseline"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


# ---------------------------------------------------------------------------
# Component definitions
# ---------------------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """
    Instantiate the TF-IDF vectorizer with project-standard configuration.

    Returns
    -------
    TfidfVectorizer
        Unfitted vectorizer ready to be trained on the corpus.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams for richer feature space
    )


def build_classifier() -> LogisticRegression:
    """
    Instantiate the Logistic Regression classifier.

    Returns
    -------
    LogisticRegression
        Unfitted classifier.
    """
    return LogisticRegression(
        max_iter=1000,        # sufficient iterations for multi-class convergence
        random_state=42,
        solver="lbfgs",       # efficient for small-to-medium, multi-class problems
    )


def save_artifacts(model: LogisticRegression, vectorizer: TfidfVectorizer) -> None:
    """
    Persist the fitted model and vectorizer to disk.

    Parameters
    ----------
    model : LogisticRegression
        Trained classifier.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[save_artifacts] Model saved      → '{MODEL_PATH}'")
    print(f"[save_artifacts] Vectorizer saved → '{VECTORIZER_PATH}'")


# ---------------------------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------------------------

def train() -> None:
    """
    End-to-end training pipeline:
        1. Load dataset
        2. Prepare stratified train/test split
        3. Fit TF-IDF vectorizer on training text
        4. Transform train and test sets
        5. Train Logistic Regression classifier
        6. Persist artifacts to disk
    """
    print("=" * 55)
    print("  AI Interview Analyzer — Baseline Training")
    print("=" * 55)

    # 1. Load dataset
    df = load_dataset(DATASET_PATH)

    # 2. Train / test split
    X_train, X_test, y_train, y_test = prepare_data(df)

    print(f"\n[train] Training samples : {len(X_train):,}")
    print(f"[train] Test samples     : {len(X_test):,}\n")

    # 3. Fit vectorizer on training corpus only (no data leakage)
    print("[train] Fitting TF-IDF vectorizer on training data …")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # 4. Transform test set using the fitted vocabulary
   
    print(
        f"[train] Vocabulary size  : {len(vectorizer.vocabulary_):,} features "
        f"(max_features=5000, ngram_range=(1,2))"
    )

    # 5. Train classifier
    print("[train] Training Logistic Regression classifier …")
    model = build_classifier()
    model.fit(X_train_tfidf, y_train)

    # 6. Save artifacts
    save_artifacts(model, vectorizer)

    print("\n[train] ✓ Training complete. Artifacts saved to 'models/baseline/'.")
    print("=" * 55)


if __name__ == "__main__":
    train()