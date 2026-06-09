"""Merge labelled interview-response sample batches into one CSV.

The source files are expected in ``data/raw`` and are merged in quality order:
Poor, Average, Good, Excellent. The final ``id`` column is regenerated so rows
are numbered consecutively from 1 through the merged dataset size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "interview_responses_batch_01.csv"

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

SOURCE_FILES = [
    ("Poor", ("poor_batch_01.csv",)),
    ("Average", ("average_batch_01.csv",)),
    ("Good", ("good_batch_01.csv", "good_batch-01.csv")),
    ("Excellent", ("excellent_batch_01.csv",)),
]


def resolve_source_file(label: str, candidates: tuple[str, ...]) -> Path:
    """Return the first existing candidate path for a source batch."""
    for filename in candidates:
        path = RAW_DATA_DIR / filename
        if path.exists():
            return path

    expected = " or ".join(candidates)
    raise FileNotFoundError(f"Missing {label} source file: expected {expected}")


def read_batch(path: Path) -> pd.DataFrame:
    """Read one source CSV and normalize its shape."""
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # Allow the script to work if a future batch includes a header row.
    if not df.empty and [str(value).strip() for value in df.iloc[0]] == COLUMNS:
        df = df.iloc[1:].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{path.name} has no rows to merge.")

    if len(df.columns) != len(COLUMNS):
        raise ValueError(
            f"{path.name} has {len(df.columns)} columns; expected {len(COLUMNS)}."
        )

    return df


def merge_batches() -> pd.DataFrame:
    """Merge all configured source batches and regenerate the id column."""
    frames = []

    for label, candidates in SOURCE_FILES:
        source_path = resolve_source_file(label, candidates)
        batch = read_batch(source_path)
        frames.append(batch)
        print(f"Loaded {len(batch):>3} {label:<9} rows from {source_path.name}")

    merged = pd.concat(frames, ignore_index=True)
    merged["id"] = range(1, len(merged) + 1)

    return merged[COLUMNS]


def main() -> int:
    try:
        merged = merge_batches()
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(OUTPUT_FILE, index=False, header=False)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nMerged {len(merged)} rows into {OUTPUT_FILE}")
    print(f"Final id range: 1 to {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
