"""
merge_datasets.py

Merge interview response datasets into a single master dataset.

Source:
    data/raw/
        - poor_batch_01.csv
        - poor_batch_02.csv
        - average_batch_01.csv
        - average_batch_02.csv
        - good_batch_01.csv
        - good_batch_02.csv
        - excellent_batch_01.csv
        - excellent_batch_02.csv

Output:
    data/raw/interview_responses.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DATA_DIR / "interview_responses.csv"


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


# Source Files

SOURCE_FILES = [
    ("Poor", "poor_batch_01.csv"),
    ("Poor", "poor_batch_02.csv"),
    ("Average", "average_batch_01.csv"),
    ("Average", "average_batch_02.csv"),
    ("Good", "good_batch_01.csv"),
    ("Good", "good_batch_02.csv"),
    ("Excellent", "excellent_batch_01.csv"),
    ("Excellent", "excellent_batch_02.csv"),
]


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

def merge_datasets() -> pd.DataFrame:
    """Merge all source datasets."""

    dataframes = []

    for _, filename in SOURCE_FILES:

        file_path = RAW_DATA_DIR / filename

        df = read_dataset(file_path)

        dataframes.append(df)

        print(
            f"✓ Loaded {len(df):>3} rows "
            f"from {filename}"
        )

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


# Main

def main() -> int:

    try:

        merged_df = merge_datasets()

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"Writing to:\n{OUTPUT_FILE.resolve()}")

        merged_df.to_csv    (
            OUTPUT_FILE,
            index=False,
        )

        print("\n" + "=" * 50)
        print("MERGE COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Total rows : {len(merged_df)}")
        print(f"Output file: {OUTPUT_FILE.name}")
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
