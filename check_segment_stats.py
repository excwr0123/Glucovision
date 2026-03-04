#!/usr/bin/env python3
"""Print segment statistics for D1Namo and AZT1D glucose series.

Segment rule:
- Sort by datetime
- New segment starts when time gap > gap_threshold_min (default 6 min)

Valid segment rule (aligned with current feature construction):
- Must produce at least one usable sample after:
  - history window (default 6 points)
  - max prediction horizon (default 24 points)
- Effective sample count per segment:
  max(0, segment_len - history_window - max_horizon)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def segment_lengths(datetimes: pd.Series, gap_threshold_min: float) -> list[int]:
    dt = pd.to_datetime(datetimes, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if dt.empty:
        return []
    delta = dt.diff().dt.total_seconds().div(60)
    seg_id = (delta.isna() | (delta > gap_threshold_min)).cumsum()
    return [int(x) for x in dt.groupby(seg_id, sort=False).size().tolist()]


def valid_samples_from_len(seg_len: int, history_window: int, max_horizon: int) -> int:
    return max(0, seg_len - history_window - max_horizon)


def compute_stats_from_lengths(lengths: list[int], history_window: int, max_horizon: int) -> dict:
    valid_counts = [valid_samples_from_len(n, history_window, max_horizon) for n in lengths]
    valid_segments = sum(1 for c in valid_counts if c > 0)
    return {
        "total_segments": len(lengths),
        "valid_segments": valid_segments,
        "invalid_segments": len(lengths) - valid_segments,
        "valid_samples": int(sum(valid_counts)),
        "max_segment_len": max(lengths) if lengths else 0,
    }


def d1namo_rows(gap_threshold_min: float, history_window: int, max_horizon: int) -> list[dict]:
    patients = ["001", "002", "004", "006", "007", "008"]
    root = Path("diabetes_subset_pictures-glucose-food-insulin")
    rows = []
    for p in patients:
        f = root / p / "glucose.csv"
        if not f.exists():
            rows.append(
                {
                    "dataset": "D1Namo",
                    "patient": p,
                    "total_rows": -1,
                    "total_segments": -1,
                    "valid_segments": -1,
                    "invalid_segments": -1,
                    "valid_samples": -1,
                    "max_segment_len": -1,
                }
            )
            continue
        df = pd.read_csv(f)
        dt = df["date"].astype(str) + " " + df["time"].astype(str)
        lengths = segment_lengths(dt, gap_threshold_min)
        stats = compute_stats_from_lengths(lengths, history_window, max_horizon)
        rows.append(
            {
                "dataset": "D1Namo",
                "patient": p,
                "total_rows": int(len(df)),
                **stats,
            }
        )
    return rows


def detect_azt1d_root() -> Path | None:
    for c in [Path("AZT1D/CGM Records"), Path("AZT1D 2025/CGM Records")]:
        if c.exists() and c.is_dir():
            return c
    return None


def azt1d_rows(gap_threshold_min: float, history_window: int, max_horizon: int) -> list[dict]:
    root = detect_azt1d_root()
    if root is None:
        return []

    rows = []
    subject_dirs = sorted([p for p in root.glob("Subject *") if p.is_dir()], key=lambda p: int(p.name.split()[-1]))
    for sd in subject_dirs:
        subject = sd.name.split()[-1]
        f = sd / f"{sd.name}.csv"
        if not f.exists():
            rows.append(
                {
                    "dataset": "AZT1D",
                    "patient": subject,
                    "total_rows": -1,
                    "total_segments": -1,
                    "valid_segments": -1,
                    "invalid_segments": -1,
                    "valid_samples": -1,
                    "max_segment_len": -1,
                }
            )
            continue

        df = pd.read_csv(f)
        if "EventDateTime" not in df.columns:
            lengths = []
        else:
            lengths = segment_lengths(df["EventDateTime"], gap_threshold_min)

        stats = compute_stats_from_lengths(lengths, history_window, max_horizon)
        rows.append(
            {
                "dataset": "AZT1D",
                "patient": subject,
                "total_rows": int(len(df)),
                **stats,
            }
        )
    return rows


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("No data found.")
        return
    df = pd.DataFrame(rows)
    try:
        df["patient_sort"] = df["patient"].astype(int)
        df = df.sort_values("patient_sort").drop(columns=["patient_sort"])
    except Exception:
        df = df.sort_values("patient")
    cols = [
        "patient",
        "total_rows",
        "total_segments",
        "valid_segments",
        "invalid_segments",
        "valid_samples",
        "max_segment_len",
    ]
    print(df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment statistics for D1Namo and AZT1D.")
    parser.add_argument("--gap-threshold-min", type=float, default=6.0, help="New segment when gap > this value (minutes).")
    parser.add_argument("--history-window", type=int, default=6, help="History window points.")
    parser.add_argument("--max-horizon", type=int, default=24, help="Max prediction horizon points.")
    args = parser.parse_args()

    print(
        f"Rule: gap > {args.gap_threshold_min:.1f} min starts new segment | "
        f"valid sample count per segment = max(0, len - {args.history_window} - {args.max_horizon})"
    )

    d_rows = d1namo_rows(args.gap_threshold_min, args.history_window, args.max_horizon)
    a_rows = azt1d_rows(args.gap_threshold_min, args.history_window, args.max_horizon)

    print_table("D1Namo Segment Stats", d_rows)
    print_table("AZT1D Segment Stats", a_rows)


if __name__ == "__main__":
    main()
