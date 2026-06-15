"""
dataset_config.py

Centralized dataset configuration and path resolution.
Supports multiple dataset categories (e.g., 'introduction', 'why-to-hire').
"""

from pathlib import Path


# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# Valid dataset categories
VALID_CATEGORIES = {"introduction", "why-to-hire"}
DEFAULT_CATEGORY = "introduction"


class DatasetConfig:
    """Manages paths and configuration for a dataset category."""

    def __init__(self, category: str = DEFAULT_CATEGORY):
        """
        Initialize configuration for a dataset category.

        Parameters
        ----------
        category : str
            Dataset category (e.g., 'introduction', 'why-to-hire').
            Defaults to 'introduction'.

        Raises
        ------
        ValueError
            If category is not recognized.
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Valid options: {sorted(VALID_CATEGORIES)}"
            )
        self.category = category
        self.category_dir = DATA_ROOT / category

    @property
    def input_dir(self) -> Path:
        """Directory containing batch CSV files for this category."""
        return self.category_dir

    @property
    def output_file(self) -> Path:
        """Path to the merged interview_responses.csv."""
        return self.category_dir / "interview_responses.csv"

    def get_batch_files(self) -> list[Path]:
        """
        Discover all batch CSV files in the category directory.
        Returns files matching pattern: *_batch_*.csv (excluding interview_responses.csv).

        Returns
        -------
        list[Path]
            Sorted list of batch file paths.
        """
        if not self.category_dir.exists():
            return []

        batch_files = [
            f for f in self.category_dir.glob("*_batch_*.csv")
            if f.name != "interview_responses.csv"
        ]
        return sorted(batch_files)

    def __repr__(self) -> str:
        return (
            f"DatasetConfig(category='{self.category}', "
            f"dir='{self.category_dir}')"
        )


def get_dataset_config(category: str = DEFAULT_CATEGORY) -> DatasetConfig:
    """
    Factory function to create a DatasetConfig.

    Parameters
    ----------
    category : str
        Dataset category. Defaults to 'introduction'.

    Returns
    -------
    DatasetConfig
        Configuration object for the category.
    """
    return DatasetConfig(category)


if __name__ == "__main__":
    # Demo: show available categories and their paths
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"Available categories: {sorted(VALID_CATEGORIES)}\n")

    for cat in sorted(VALID_CATEGORIES):
        cfg = get_dataset_config(cat)
        print(f"{cfg}")
        print(f"  Input dir:       {cfg.input_dir}")
        print(f"  Output file:     {cfg.output_file}")
        batches = cfg.get_batch_files()
        print(f"  Batch files:     {len(batches)} found")
        if batches:
            for bf in batches[:3]:
                print(f"    - {bf.name}")
            if len(batches) > 3:
                print(f"    ... and {len(batches) - 3} more")
        print()
