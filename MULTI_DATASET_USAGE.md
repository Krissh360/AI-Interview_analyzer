# Multi-Dataset Support Guide

This project now supports working with multiple dataset categories seamlessly without modifying source code paths. All Python scripts use command-line arguments to specify which dataset category to work with.

## Supported Categories

- `introduction` (default)
- `why-to-hire`

## Batch File Naming Convention

Batch files must follow the pattern: `{label}_batch_{number}.csv`

Examples:
- `poor_batch_01.csv`, `poor_batch_1.csv`, `poor_batch_a.csv` → all valid
- `average_batch_01.csv`, `average_batch_02.csv` → multiple per tier
- `good_batch_1.csv` → single file per tier
- `excellent_batch_001.csv` → any numbering scheme

**The system auto-discovers all files matching `*_batch_*.csv` pattern, so you can have 1, 2, 3, or any number of batch files per quality tier.**

## Directory Structure

```
data/raw/
├── introduction/
│   ├── poor_batch_01.csv
│   ├── poor_batch_02.csv
│   ├── poor_batch_03.csv
│   ├── average_batch_01.csv
│   ├── average_batch_02.csv
│   ├── average_batch_03.csv
│   ├── good_batch_01.csv
│   ├── good_batch_02.csv
│   ├── good_batch_03.csv
│   ├── excellent_batch_01.csv
│   ├── excellent_batch_02.csv
│   ├── excellent_batch_03.csv
│   └── interview_responses.csv (merged output)
└── why-to-hire/
    ├── poor_batch_01.csv
    ├── poor_batch_02.csv
    ├── poor_batch_03.csv
    ├── average_batch_01.csv
    ├── average_batch_02.csv
    ├── average_batch_03.csv
    ├── good_batch_01.csv
    ├── good_batch_02.csv
    ├── good_batch_03.csv
    ├── excellent_batch_01.csv
    ├── excellent_batch_02.csv
    ├── excellent_batch_03.csv
    └── interview_responses.csv (merged output)
```

## Usage

### 1. Merge Dataset Batches

**Default (introduction category):**
```bash
cd src/dataset
python merge_datasets.py
```

**For why-to-hire category:**
```bash
cd src/dataset
python merge_datasets.py --category why-to-hire
```

The script automatically discovers all `*_batch_*.csv` files in the category folder and merges them into `interview_responses.csv`.

---

### 2. Validate Dataset

**Default (introduction category):**
```bash
cd src/dataset
python validate_dataset.py
```

**For why-to-hire category:**
```bash
cd src/dataset
python validate_dataset.py --category why-to-hire
```

Validates dataset completeness, label distribution, score ranges, and consistency.

---

### 3. Preprocess Data

**Default (introduction category):**
```bash
cd src/training
python preprocess.py
```

**For why-to-hire category:**
```bash
cd src/training
python preprocess.py --category why-to-hire
```

Shows label distribution and train/test split summary for the category.

---

### 4. Train Baseline Model

**Default (introduction category):**
```bash
cd src/training
python train_baseline.py
```

**For why-to-hire category:**
```bash
cd src/training
python train_baseline.py --category why-to-hire
```

Trains TF-IDF + Logistic Regression and saves to `models/baseline/`.

---

### 5. Evaluate Baseline Model

**Default (introduction category):**
```bash
cd src/training
python evaluate_baseline.py
```

**For why-to-hire category:**
```bash
cd src/training
python evaluate_baseline.py --category why-to-hire
```

Reports accuracy, precision/recall/F1, and confusion matrix for the category.

---

## Example Workflow

### Process "introduction" category:
```bash
# Merge batches
python src/dataset/merge_datasets.py

# Validate
python src/dataset/validate_dataset.py

# Train
python src/training/train_baseline.py

# Evaluate
python src/training/evaluate_baseline.py
```

### Process "why-to-hire" category:
```bash
# Merge batches
python src/dataset/merge_datasets.py --category why-to-hire

# Validate
python src/dataset/validate_dataset.py --category why-to-hire

# Train
python src/training/train_baseline.py --category why-to-hire

# Evaluate
python src/training/evaluate_baseline.py --category why-to-hire
```

---

## Adding a New Dataset Category

1. Create a new folder under `data/raw/`:
   ```
   data/raw/new-category/
   ```

2. Add your batch CSV files:
   ```
   data/raw/new-category/
   ├── poor_batch_01.csv
   ├── average_batch_01.csv
   ├── good_batch_01.csv
   ├── excellent_batch_01.csv
   └── ...
   ```

3. Update `src/dataset/dataset_config.py` - Add the new category to `VALID_CATEGORIES`:
   ```python
   VALID_CATEGORIES = {"introduction", "why-to-hire", "new-category"}
   ```

4. Use the new category with any script:
   ```bash
   python merge_datasets.py --category new-category
   ```

---

## Configuration Module

All path resolution is centralized in `src/dataset/dataset_config.py`. This module:

- Validates category names
- Auto-discovers batch files matching `*_batch_*.csv` pattern
- Resolves input directories and output paths dynamically

You can also inspect available categories programmatically:
```bash
python src/dataset/dataset_config.py
```

This will display all configured categories and their paths.
