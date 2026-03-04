#!/usr/bin/env python3
"""Check D1Namo glucose quality per patient.

Outputs a compact table:
- patient
- total_rows
- duplicate_time
- max_gap_min
- rows_gt_5min
- rows_gt_5min_list
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PATIENTS_D1NAMO = ["001", "002", "004", "006", "007", "008"]
DATA_ROOT = Path("diabetes_subset_pictures-glucose-food-insulin")


def load_glucose(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def analyze_patient(df: pd.DataFrame) -> dict:
    dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    valid = pd.DataFrame({"datetime": dt}).dropna(subset=["datetime"]).copy()
    valid = valid.sort_values("datetime").reset_index(drop=True)

    if len(valid) < 2:
        return {
            "total_rows": int(len(df)),
            "duplicate_time": 0,
            "max_gap_min": 0.0,
            "monitoring_days": 0.0,
            "rows_gt_5min": 0,
            "rows_gt_5min_list": [],
            "has_missing_glucose": int(df["glucose"].isna().sum()) > 0 if "glucose" in df.columns else True,
        }

    valid["gap_min"] = valid["datetime"].diff().dt.total_seconds().div(60)
    intervals = valid["gap_min"].dropna()
    duplicates = int(valid["datetime"].duplicated().sum())
    gt5_series = intervals[intervals > 5.0].round(3)
    rows_gt_5min = int(len(gt5_series))
    return {
        "total_rows": int(len(df)),
        "duplicate_time": duplicates,
        "max_gap_min": round(float(intervals.max()), 3),
        "monitoring_days": round((valid["datetime"].iloc[-1] - valid["datetime"].iloc[0]).total_seconds() / 86400.0, 3),
        "rows_gt_5min": rows_gt_5min,
        "rows_gt_5min_list": gt5_series.tolist(),
        "has_missing_glucose": int(df["glucose"].isna().sum()) > 0 if "glucose" in df.columns else True,
    }


def main() -> None:
    summary_rows = []

    print("\n=== D1Namo Glucose Check (Compact) ===")

    for patient in PATIENTS_D1NAMO:
        path = DATA_ROOT / patient / "glucose.csv"
        if not path.exists():
            print(f"\n[Patient {patient}] Missing file: {path}")
            summary_rows.append(
                {
                    "patient": patient,
                    "total_rows": -1,
                    "duplicate_time": -1,
                    "max_gap_min": -1.0,
                    "monitoring_days": -1.0,
                    "rows_gt_5min": -1,
                    "rows_gt_5min_list": [],
                    "has_missing_glucose": True,
                }
            )
            continue

        df = load_glucose(path)
        summary = analyze_patient(df)
        summary["patient"] = patient
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values("patient")
    summary_cols = [
        "patient",
        "total_rows",
        "rows_gt_5min",
        "rows_gt_5min_list",
        "has_missing_glucose",
        "duplicate_time",
        "max_gap_min",
        "monitoring_days",
    ]
    print("\n=== Summary Table ===")
    print(summary_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
