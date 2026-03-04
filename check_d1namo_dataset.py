#!/usr/bin/env python3
"""Validate and profile D1Namo patient files.

Usage:
  python check_d1namo_dataset.py
  python check_d1namo_dataset.py --gap-threshold 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PATIENTS_D1NAMO = ["001", "002", "004", "006", "007", "008"]
DATA_ROOT = Path("diabetes_subset_pictures-glucose-food-insulin")
PIXTRAL_ROOT = Path("food_data/pixtral-large-latest")


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Failed reading {path}: {exc}")
        return None


def check_glucose_intervals(glucose_df: pd.DataFrame, gap_threshold: float) -> dict[str, float]:
    dt = pd.to_datetime(glucose_df["date"] + " " + glucose_df["time"], errors="coerce")
    dt = dt.dropna().sort_values().reset_index(drop=True)
    if len(dt) < 2:
        return {
            "intervals": 0,
            "strict_5min": 0,
            "tol_5min": 0,
            "strict_5min_pct": 0.0,
            "tol_5min_pct": 0.0,
            "min_gap_min": 0.0,
            "median_gap_min": 0.0,
            "max_gap_min": 0.0,
            "large_gaps": 0,
            "duplicate_timestamps": 0,
        }

    delta = dt.diff().dropna().dt.total_seconds() / 60.0
    strict_5 = int((delta == 5).sum())
    tol_5 = int(((delta >= 4.5) & (delta <= 5.5)).sum())
    total = len(delta)
    dup_count = int(dt.duplicated().sum())

    return {
        "intervals": total,
        "strict_5min": strict_5,
        "tol_5min": tol_5,
        "strict_5min_pct": round(strict_5 * 100.0 / total, 2),
        "tol_5min_pct": round(tol_5 * 100.0 / total, 2),
        "min_gap_min": round(float(delta.min()), 3),
        "median_gap_min": round(float(delta.median()), 3),
        "max_gap_min": round(float(delta.max()), 3),
        "large_gaps": int((delta > gap_threshold).sum()),
        "duplicate_timestamps": dup_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check D1Namo patient datasets.")
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=10.0,
        help="Mark glucose interval gaps larger than this threshold in minutes.",
    )
    args = parser.parse_args()

    summary_rows = []
    missing_files = []

    for patient in PATIENTS_D1NAMO:
        pdir = DATA_ROOT / patient
        glucose_path = pdir / "glucose.csv"
        insulin_path = pdir / "insulin.csv"
        food_path = pdir / "food.csv"
        pixtral_path = PIXTRAL_ROOT / f"{patient}.csv"
        pictures_dir = pdir / "food_pictures"

        glucose_df = safe_read_csv(glucose_path)
        insulin_df = safe_read_csv(insulin_path)
        food_df = safe_read_csv(food_path)
        pixtral_df = safe_read_csv(pixtral_path)

        for fp in [glucose_path, insulin_path, food_path, pixtral_path]:
            if not fp.exists():
                missing_files.append(str(fp))

        n_images = len(list(pictures_dir.glob("*.jpg"))) if pictures_dir.exists() else 0

        row = {
            "patient": patient,
            "glucose_rows": len(glucose_df) if glucose_df is not None else -1,
            "insulin_rows": len(insulin_df) if insulin_df is not None else -1,
            "food_rows": len(food_df) if food_df is not None else -1,
            "pixtral_rows": len(pixtral_df) if pixtral_df is not None else -1,
            "food_images": n_images,
        }

        if glucose_df is not None and {"date", "time"}.issubset(glucose_df.columns):
            row.update(check_glucose_intervals(glucose_df, args.gap_threshold))
        else:
            row.update(
                {
                    "intervals": -1,
                    "strict_5min": -1,
                    "tol_5min": -1,
                    "strict_5min_pct": -1.0,
                    "tol_5min_pct": -1.0,
                    "min_gap_min": -1.0,
                    "median_gap_min": -1.0,
                    "max_gap_min": -1.0,
                    "large_gaps": -1,
                    "duplicate_timestamps": -1,
                }
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("patient")

    print("\n=== D1Namo Dataset Check Summary ===")
    print(f"Gap threshold: {args.gap_threshold:.1f} min")
    print(summary_df.to_string(index=False))

    if missing_files:
        print("\n[WARN] Missing files:")
        for fp in missing_files:
            print(f"  - {fp}")
    else:
        print("\nAll expected files were found.")

    print("\n=== Raw Column Overview (patient 001 as example) ===")
    sample_patient = "001"
    sample_files = {
        "glucose.csv": DATA_ROOT / sample_patient / "glucose.csv",
        "insulin.csv": DATA_ROOT / sample_patient / "insulin.csv",
        "food.csv": DATA_ROOT / sample_patient / "food.csv",
        "pixtral.csv": PIXTRAL_ROOT / f"{sample_patient}.csv",
    }
    for name, path in sample_files.items():
        df = safe_read_csv(path)
        if df is None:
            print(f"{name:<12} -> [missing]")
            continue
        print(f"{name:<12} -> columns={list(df.columns)}")


if __name__ == "__main__":
    main()
