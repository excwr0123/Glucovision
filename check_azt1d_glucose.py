#!/usr/bin/env python3
"""Check AZT1D glucose quality per subject.

Summary columns:
- patient
- total_rows
- rows_gt_5min
- has_missing_glucose
- duplicate_time
- max_gap_min
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_CANDIDATES = [
    Path("AZT1D/CGM Records"),
    Path("AZT1D 2025/CGM Records"),
]
GLUCOSE_COL_CANDIDATES = ["CGM", "Readings (CGM / BGM)", "Readings (CGM/BGM)"]
DATETIME_COL = "EventDateTime"


def detect_root() -> Path | None:
    for root in ROOT_CANDIDATES:
        if root.exists() and root.is_dir():
            return root
    return None


def pick_glucose_col(columns: list[str]) -> str | None:
    for c in GLUCOSE_COL_CANDIDATES:
        if c in columns:
            return c
    return None


def analyze_subject(df: pd.DataFrame, glucose_col: str | None) -> dict:
    total_rows = int(len(df))
    if DATETIME_COL not in df.columns:
        return {
            "total_rows": total_rows,
            "rows_gt_5min": -1,
            "has_missing_glucose": True,
            "duplicate_time": -1,
            "max_gap_min": -1.0,
            "monitoring_days": -1.0,
        }

    has_missing_glucose = True if glucose_col is None else bool(df[glucose_col].isna().any())
    dt = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    valid = pd.DataFrame({"datetime": dt}).dropna(subset=["datetime"]).copy()
    valid = valid.sort_values("datetime").reset_index(drop=True)

    if len(valid) < 2:
        return {
            "total_rows": total_rows,
            "rows_gt_5min": 0,
            "has_missing_glucose": has_missing_glucose,
            "duplicate_time": 0,
            "max_gap_min": 0.0,
            "monitoring_days": 0.0,
        }

    valid["gap_min"] = valid["datetime"].diff().dt.total_seconds().div(60)
    intervals = valid["gap_min"].dropna()
    gt5_series = intervals[intervals > 5.0].round(3)

    return {
        "total_rows": total_rows,
        "rows_gt_5min": int(len(gt5_series)),
        "has_missing_glucose": has_missing_glucose,
        "duplicate_time": int(valid["datetime"].duplicated().sum()),
        "max_gap_min": round(float(intervals.max()), 3),
        "monitoring_days": round((valid["datetime"].iloc[-1] - valid["datetime"].iloc[0]).total_seconds() / 86400.0, 3),
    }


def main() -> None:
    root = detect_root()
    if root is None:
        print("No AZT1D dataset root found. Tried:")
        for c in ROOT_CANDIDATES:
            print(f"  - {c}")
        return

    subject_dirs = sorted([p for p in root.glob("Subject *") if p.is_dir()], key=lambda p: int(p.name.split()[-1]))
    summary_rows: list[dict] = []

    print("\n=== AZT1D Glucose Check (Compact) ===")
    print(f"Using dataset root: {root}")

    for subject_dir in subject_dirs:
        subject_num = subject_dir.name.split()[-1]
        csv_path = subject_dir / f"{subject_dir.name}.csv"

        if not csv_path.exists():
            summary_rows.append(
                {
                    "patient": subject_num,
                    "total_rows": -1,
                    "rows_gt_5min": -1,
                    "has_missing_glucose": True,
                    "duplicate_time": -1,
                    "max_gap_min": -1.0,
                    "monitoring_days": -1.0,
                }
            )
            continue

        df = pd.read_csv(csv_path)
        glucose_col = pick_glucose_col(df.columns.tolist())
        summary = analyze_subject(df, glucose_col)
        summary["patient"] = subject_num
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df["patient_sort"] = summary_df["patient"].astype(int)
    summary_df = summary_df.sort_values("patient_sort").drop(columns=["patient_sort"])

    summary_cols = [
        "patient",
        "total_rows",
        "rows_gt_5min",
        "has_missing_glucose",
        "duplicate_time",
        "max_gap_min",
        "monitoring_days",
    ]
    print("\n=== Summary Table ===")
    print(summary_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
