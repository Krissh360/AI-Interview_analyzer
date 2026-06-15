"""
merge_datasets.py

Merge interview response datasets into a single master dataset.
Automatic batch file discovery per dataset category.

Usage
-----
    python merge_datasets.py                           # uses 'introduction' (default)
    python merge_datasets.py --category introduction   # explicit category
    python merge_datasets.py --category why-to-hire    # another category

Source:
    data/raw/<category>/
        - *_batch_*.csv (auto-discovered)

Output:
    data/raw/<category>/interview_responses.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from dataset_config import DatasetConfig, DEFAULT_CATEGORY, VALID_CATEGORIES


# Dataset Schema

COLUMNS = [
    "id",
    "answer",
    "content_score",
    "relevance_score",
    "vocabulary_score",
    "structure_score",
    "overall_score",
    "label",
]

SCORE_WEIGHTS = {
    "content_score": 0.35,
    "relevance_score": 0.25,
    "vocabulary_score": 0.25,
    "structure_score": 0.15,
}


# Validation

def validate_schema(df: pd.DataFrame, filename: str) -> None:
    """Validate required columns exist."""

    missing_columns = set(COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{filename} is missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )


# Score Calculation

def calculate_overall_score(row: pd.Series) -> float:
    """Calculate weighted overall score from component scores."""

    return round(
        sum(row[column] * weight for column, weight in SCORE_WEIGHTS.items()),
        2,
    )


def get_label(score: float) -> str:
    """Derive dataset label from overall score."""

    if score < 2.0:
        return "Poor"
    if score <= 3.0:
        return "Average"
    if score <= 4.0:
        return "Good"
    return "Excellent"


def recalculate_scores_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate overall score and label before merging."""

    df = df.copy()

    for column in SCORE_WEIGHTS:
        df[column] = pd.to_numeric(df[column])

    df["overall_score"] = df.apply(
        calculate_overall_score,
        axis=1,
    )
    df["label"] = df["overall_score"].apply(get_label)

    return df


# Read Dataset

def read_dataset(path: Path) -> pd.DataFrame:
    """Read and validate a dataset."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path.name} contains no rows.")

    validate_schema(df, path.name)

    df = recalculate_scores_and_labels(df)

    return df[COLUMNS]


# Merge Logic

def merge_datasets(config: DatasetConfig) -> pd.DataFrame:
    """Merge all source datasets for the given category."""

    batch_files = config.get_batch_files()

    if not batch_files:
        raise FileNotFoundError(
            f"No batch files found in {config.input_dir}. "
            f"Expected files matching '*_batch_*.csv'."
        )

    dataframes = []

    for file_path in batch_files:
        df = read_dataset(file_path)
        dataframes.append(df)
        print(f"✓ Loaded {len(df):>3} rows from {file_path.name}")

    merged_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    # Regenerate IDs
    merged_df["id"] = range(
        1,
        len(merged_df) + 1,
    )

    return merged_df[COLUMNS]


# CLI & Main

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge interview response batch datasets into a single file."
    )
    parser.add_argument(
        "--category",
        type=str,
        default=DEFAULT_CATEGORY,
        choices=sorted(VALID_CATEGORIES),
        help=f"Dataset category (default: {DEFAULT_CATEGORY})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = DatasetConfig(args.category)
        print(f"Using category: {config.category}")
        print(f"Input dir:     {config.input_dir}\n")

        merged_df = merge_datasets(config)

        config.output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"\nWriting to:\n{config.output_file.resolve()}")

        merged_df.to_csv(
            config.output_file,
            index=False,
        )

        print("\n" + "=" * 50)
        print("MERGE COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Category   : {config.category}")
        print(f"Total rows : {len(merged_df)}")
        print(f"Output file: {config.output_file.name}")
        print(f"ID range   : 1 - {len(merged_df)}")
        print("=" * 50)

        return 0

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
