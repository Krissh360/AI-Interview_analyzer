import pandas as pd
import numpy as np
import sys

CSV_PATH = "interview_responses_batch_01.csv"
SCORE_COLS = ["content_score", "relevance_score", "vocabulary_score", "structure_score", "overall_score"]
WEIGHTS = {
    "content_score": 0.35,
    "relevance_score": 0.25,
    "vocabulary_score": 0.25,
    "structure_score": 0.15,
}

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

def compute_overall(row):
    return round(
        sum(row[col] * w for col, w in WEIGHTS.items()), 2
    )

def compute_label(score):
    if score < 2.0:
        return "Poor"
    elif score <= 3.0:
        return "Average"
    elif score <= 4.0:
        return "Good"
    else:
        return "Excellent"

def validate():
    # Load Dataset
    section("LOADING DATASET")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"  File      : {CSV_PATH}")
        print(f"  Rows      : {len(df)}")
        print(f"  Columns   : {list(df.columns)}")
    except FileNotFoundError:
        print(f"  {FAIL} File '{CSV_PATH}' not found. Exiting.")
        sys.exit(1)

    all_passed = True

    # 1. Missing Values
    section("1 · MISSING VALUES")
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    for col, count in null_counts.items():
        status = PASS if count == 0 else FAIL
        print(f"  {status}  {col:<25} nulls: {count}")

    if total_nulls > 0:
        all_passed = False
        print(f"\n  {FAIL} Total missing values: {total_nulls}")
    else:
        print(f"\n  {PASS} No missing values found.")

    # 2. Label Distribution 
    section("2 · LABEL DISTRIBUTION")
    expected = {"Poor": 25, "Average": 25, "Good": 25, "Excellent": 25}
    actual = df["label"].value_counts().to_dict()

    print(f"  {'Label':<12} {'Expected':>10} {'Actual':>10}  Status")
    print(f"  {'-'*46}")
    for label, exp_count in expected.items():
        actual_count = actual.get(label, 0)
        status = PASS if actual_count == exp_count else FAIL
        if status == FAIL:
            all_passed = False
        print(f"  {label:<12} {exp_count:>10} {actual_count:>10}  {status}")

    unexpected = set(actual.keys()) - set(expected.keys())
    if unexpected:
        all_passed = False
        print(f"\n  {FAIL} Unexpected labels found: {unexpected}")
    else:
        print(f"\n  {PASS} No unexpected labels.")

    # 3. Score Ranges 
    section("3 · SCORE RANGES  (expected: 1 – 5 for component scores)")
    component_cols = ["content_score", "relevance_score", "vocabulary_score", "structure_score"]
    range_issues = []

    for col in component_cols:
        if col not in df.columns:
            print(f"  {FAIL}  Column '{col}' not found.")
            all_passed = False
            continue
        out = df[(df[col] < 1) | (df[col] > 5)]
        status = PASS if out.empty else FAIL
        if not out.empty:
            all_passed = False
            range_issues.append((col, out.index.tolist()))
        print(f"  {status}  {col:<25} min={df[col].min()}  max={df[col].max()}")

    # overall_score can exceed 5 only if component scores are 5; just check >= 1
    if "overall_score" in df.columns:
        out_os = df[(df["overall_score"] < 1) | (df["overall_score"] > 5)]
        status = PASS if out_os.empty else WARN
        print(f"  {status}  {'overall_score':<25} min={df['overall_score'].min():.2f}  max={df['overall_score'].max():.2f}")
        if not out_os.empty:
            print(f"          Rows outside [1,5]: {out_os.index.tolist()}")

    if range_issues:
        print(f"\n  {FAIL} Out-of-range rows:")
        for col, rows in range_issues:
            print(f"       {col}: rows {rows}")
    else:
        print(f"\n  {PASS} All component scores within valid range.")

    # 4. Label & Overall Score Consistency
    section("4 · LABEL & OVERALL SCORE CONSISTENCY")

    required = list(WEIGHTS.keys()) + ["overall_score", "label"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"  {FAIL} Missing columns for consistency check: {missing_cols}")
        all_passed = False
    else:
        df["_calc_overall"] = df.apply(compute_overall, axis=1)
        df["_calc_label"]   = df["_calc_overall"].apply(compute_label)

        # overall_score tolerance: floating-point rounding ±0.01
        overall_mismatch = df[abs(df["overall_score"] - df["_calc_overall"]) > 0.01]
        label_mismatch   = df[df["label"] != df["_calc_label"]]

        if overall_mismatch.empty:
            print(f"  {PASS} overall_score matches formula for all rows.")
        else:
            all_passed = False
            print(f"  {FAIL} overall_score mismatch in {len(overall_mismatch)} row(s):")
            for _, row in overall_mismatch.iterrows():
                print(f"       id={int(row['id'])}  stored={row['overall_score']}  calculated={row['_calc_overall']}")

        if label_mismatch.empty:
            print(f"  {PASS} Labels match calculated overall_score for all rows.")
        else:
            all_passed = False
            print(f"  {FAIL} Label mismatch in {len(label_mismatch)} row(s):")
            for _, row in label_mismatch.iterrows():
                print(f"       id={int(row['id'])}  stored='{row['label']}'  expected='{row['_calc_label']}'  overall={row['overall_score']}")

        df.drop(columns=["_calc_overall", "_calc_label"], inplace=True)

    # 5. Duplicate Detection 
    section("5 · DUPLICATE DETECTION")
    dup_mask = df["answer"].duplicated(keep=False)
    duplicates = df[dup_mask]

    if duplicates.empty:
        print(f"  {PASS} No duplicate answers found.")
    else:
        all_passed = False
        print(f"  {FAIL} {len(duplicates)} duplicate answer(s) detected:")
        for _, row in duplicates.iterrows():
            preview = str(row["answer"])[:80].replace("\n", " ")
            print(f"       id={int(row['id'])}  \"{preview}...\"")

    # Summary
    section("VALIDATION SUMMARY")
    if all_passed:
        print(f"  {PASS} All checks passed. Dataset looks clean.\n")
    else:
        print(f"  {FAIL} One or more checks failed. Review issues above.\n")

    return all_passed


if __name__ == "__main__":
    result = validate()
    sys.exit(0 if result else 1)