"""
merge_datasets.py

Merge interview response datasets into a single master dataset.

Source:
    data/raw/
        - poor_batch_01.csv
        - average_batch_01.csv
        - good_batch_01.csv
        - excellent_batch_01.csv

Output:
    data/raw/interview_responses_batch_01.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = RAW_DATA_DIR / "interview_responses_batch_01.csv"


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


# Source Files

SOURCE_FILES = [
    ("Poor", "poor_batch_01.csv"),
    ("Average", "average_batch_01.csv"),
    ("Good", "good_batch_01.csv"),
    ("Excellent", "excellent_batch_01.csv"),
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


# Read Dataset

def read_dataset(path: Path) -> pd.DataFrame:
    """Read and validate a dataset."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path.name} contains no rows.")

    validate_schema(df, path.name)

    return df[COLUMNS]


# Merge Logic

def merge_datasets() -> pd.DataFrame:
    """Merge all source datasets."""

    dataframes = []

    for label, filename in SOURCE_FILES:

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