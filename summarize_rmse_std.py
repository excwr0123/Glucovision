#!/usr/bin/env python3
"""Summarize RMSE mean/std and compare two result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_stats(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    g = df.groupby(["Prediction Horizon", "Approach"], as_index=False).agg(
        rmse_mean=("RMSE", "mean"),
        rmse_std=("RMSE", "std"),
        n=("RMSE", "count"),
    )
    g = g.rename(
        columns={
            "rmse_mean": f"{tag}_rmse_mean",
            "rmse_std": f"{tag}_rmse_std",
            "n": f"{tag}_n",
        }
    )
    g[f"{tag}_rmse_pm_std"] = g[f"{tag}_rmse_mean"].map(lambda x: f"{x:.3f}") + "±" + g[f"{tag}_rmse_std"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "nan"
    )
    return g


def print_pivot(stats: pd.DataFrame, col: str, title: str) -> None:
    piv = (
        stats.pivot(index="Prediction Horizon", columns="Approach", values=col)
        .reindex(columns=["Glucose+Insulin", "LastMeal", "Bezier"])
        .sort_index()
    )
    print(f"\n=== {title} ===")
    print(piv.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="results/d1namo_comparison.csv", help="Old/original result CSV path")
    parser.add_argument(
        "--new",
        default="results/d1namo_comparison_segment_7030.csv",
        help="New/segment-aware result CSV path",
    )
    parser.add_argument("--out", default="results/d1namo_rmse_std_comparison_segment_7030.csv", help="Output CSV path")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    out_path = Path(args.out)

    if not old_path.exists():
        raise FileNotFoundError(f"Old file not found: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New file not found: {new_path}")

    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    old_stats = build_stats(old_df, "old")
    new_stats = build_stats(new_df, "new")

    merged = pd.merge(
        old_stats[
            ["Prediction Horizon", "Approach", "old_rmse_mean", "old_rmse_std", "old_n", "old_rmse_pm_std"]
        ],
        new_stats[
            ["Prediction Horizon", "Approach", "new_rmse_mean", "new_rmse_std", "new_n", "new_rmse_pm_std"]
        ],
        on=["Prediction Horizon", "Approach"],
        how="outer",
    ).sort_values(["Prediction Horizon", "Approach"])
    merged["delta_mean_new_minus_old"] = merged["new_rmse_mean"] - merged["old_rmse_mean"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print(f"Old file: {old_path}")
    print(f"New file: {new_path}")
    print(f"Saved comparison CSV: {out_path}")

    print_pivot(old_stats, "old_rmse_pm_std", "Original RMSE±Std")
    print_pivot(new_stats, "new_rmse_pm_std", "Segment-7030 RMSE±Std")

    delta_piv = (
        merged.pivot(index="Prediction Horizon", columns="Approach", values="delta_mean_new_minus_old")
        .reindex(columns=["Glucose+Insulin", "LastMeal", "Bezier"])
        .sort_index()
    )
    print("\n=== Delta Mean (new - old, lower is better) ===")
    print(delta_piv.round(3).to_string())


if __name__ == "__main__":
    main()
