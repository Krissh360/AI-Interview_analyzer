"""
preprocess.py
-------------
Data loading and preprocessing utilities for the AI Interview Analyzer.
Handles CSV ingestion, column validation, and train/test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from ..dataset.dataset_config import DatasetConfig, DEFAULT_CATEGORY, VALID_CATEGORIES


# Columns required to exist in the dataset
REQUIRED_COLUMNS = [
    "id",
    "answer",
    "content_score",
    "relevance_score",
    "vocabulary_score",
    "structure_score",
    "overall_score",
    "label",
]

# Valid target class labels
VALID_LABELS = {"Poor", "Average", "Good", "Excellent"}


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the interview responses CSV and validate its schema.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g. 'data/raw/introduction/interview_responses.csv').

    Returns
    -------
    pd.DataFrame
        Validated dataframe ready for downstream processing.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing or label values are unexpected.
    """
    # --- Load ---
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at path: '{filepath}'")

    print(f"[load_dataset] Loaded {len(df):,} rows from '{filepath}'.")

    # --- Column validation ---
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_cols)}"
        )

    # --- Drop rows with null answers or labels ---
    before = len(df)
    df = df.dropna(subset=["answer", "label"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[load_dataset] Dropped {dropped} rows with null 'answer' or 'label'.")

    # --- Label validation ---
    unexpected_labels = set(df["label"].unique()) - VALID_LABELS
    if unexpected_labels:
        raise ValueError(
            f"Unexpected label values found: {unexpected_labels}. "
            f"Expected one of: {VALID_LABELS}"
        )

    # --- Ensure answer column is string type ---
    df["answer"] = df["answer"].astype(str).str.strip()

    print(f"[load_dataset] Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


def prepare_data(
    df: pd.DataFrame,
    text_col: str = "answer",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split the dataframe into stratified train / test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataframe returned by `load_dataset`.
    text_col : str
        Name of the column containing raw text (default: 'answer').
    label_col : str
        Name of the target column (default: 'label').
    test_size : float
        Proportion of data to reserve for testing (default: 0.2).
    random_state : int
        Seed for reproducibility (default: 42).

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
        X_* are pandas Series of raw text strings.
        y_* are pandas Series of label strings.
    """
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(
        f"[prepare_data] Split complete — "
        f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples."
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess interview responses dataset."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_CATEGORY,
        choices=sorted(VALID_CATEGORIES),
        help=f"Dataset category (default: {DEFAULT_CATEGORY})",
    )
    args = parser.parse_args()

    config = DatasetConfig(args.category)
    DATA_PATH = str(config.output_file)

    df = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = prepare_data(df)

    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Category: {args.category}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())