"""
train_baseline.py
-----------------
Trains a TF-IDF + Logistic Regression baseline classifier for the
AI Interview Analyzer. Saves the fitted vectorizer and model to disk
using joblib for later evaluation or serving.

Usage
-----
    python -m src.training.train_baseline
    python -m src.training.train_baseline --category why-to-hire
    python src/training/train_baseline.py
    python src/training/train_baseline.py --category why-to-hire
"""

import argparse
import os
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# Allow running the script directly from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.preprocess import load_dataset, prepare_data
from src.dataset.dataset_config import DatasetConfig, DEFAULT_CATEGORY, VALID_CATEGORIES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Note: These are templates and will be resolved at runtime based on --category

DEFAULT_DATASET_PATH = "../../data/raw/introduction/interview_responses.csv"
DEFAULT_MODEL_DIR = "../../models/baseline"
MODEL_PATH_TEMPLATE = "MODEL_PATH"  # Will be set in main()
VECTORIZER_PATH_TEMPLATE = "VECTORIZER_PATH"  # Will be set in main()


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


def save_artifacts(model: LogisticRegression, vectorizer: TfidfVectorizer, model_dir: str) -> None:
    """
    Persist the fitted model and vectorizer to disk.

    Parameters
    ----------
    model : LogisticRegression
        Trained classifier.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    model_dir : str
        Directory to save artifacts to.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"[save_artifacts] Model saved      → '{model_path}'")
    print(f"[save_artifacts] Vectorizer saved → '{vectorizer_path}'")


# ---------------------------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------------------------

def train(dataset_path: str, model_dir: str) -> None:
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
    df = load_dataset(dataset_path)

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
    save_artifacts(model, vectorizer, model_dir)

    print("\n[train] ✓ Training complete. Artifacts saved to 'models/baseline/'.")
    print("=" * 55)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train baseline TF-IDF + Logistic Regression classifier."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_CATEGORY,
        choices=sorted(VALID_CATEGORIES),
        help=f"Dataset category (default: {DEFAULT_CATEGORY})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = DatasetConfig(args.category)
    
    dataset_path = str(config.output_file)
    model_dir = "../../models/baseline"
    
    print(f"Category: {args.category}\n")
    train(dataset_path, model_dir)