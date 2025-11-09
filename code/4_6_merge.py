"""
Pivot judge responses from `4_5_batch_output.csv` into a single wide CSV for analysis.

This script is intentionally simple: it reads the long-form judge rows produced by
`4_5_batch_results_csv.py`, pivots answers/text per judge type, and writes the
result to `results/4_6_batch_merge.csv`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Resolve project locations regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
INPUT_CSV = RESULTS_DIR / "4_5_batch_output.csv"
OUTPUT_CSV = RESULTS_DIR / "4_6_batch_merge.csv"


def load_judge_results() -> pd.DataFrame:
    """Read the long-form judge CSV and ensure the key columns exist."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot locate {INPUT_CSV}. Run 4_5_batch_results_csv.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    required_cols = {"custom_id", "permutation_id", "judge_type", "judge_answer", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV missing required columns: "
            + ", ".join(sorted(missing))
        )
    return df


def pivot_results(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot rows so each permutation lives on a single line."""
    pivot_df = (
        df.pivot_table(
            index=["permutation_id"],
            columns="judge_type",
            values=["judge_answer", "text"],
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )

    pivot_df.columns = [
        f"{value}_{judge_type}" if judge_type else value
        for value, judge_type in pivot_df.columns
    ]
    return pivot_df


def write_output(df: pd.DataFrame) -> Path:
    """Persist the merged CSV, falling back if Excel locks the target file."""
    try:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Wrote merged judge results to {OUTPUT_CSV}")
        return OUTPUT_CSV
    except PermissionError:
        fallback = OUTPUT_CSV.with_name(f"{OUTPUT_CSV.stem}_new{OUTPUT_CSV.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8")
        print(
            "Target CSV appears to be open (PermissionError). "
            f"Results written to fallback file: {fallback}"
        )
        return fallback


def main() -> None:
    judge_results = load_judge_results()
    merged = pivot_results(judge_results)
    write_output(merged)


if __name__ == "__main__":
    main()
