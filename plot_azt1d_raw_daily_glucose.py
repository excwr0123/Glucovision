#!/usr/bin/env python3
"""Plot AZT1D raw daily glucose traces per patient, with meal/insulin events."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_CANDIDATES = [Path("AZT1D/CGM Records"), Path("AZT1D 2025/CGM Records")]
OUT_DIR = Path("results/plots/azt1d_raw_daily_glucose")
GLUCOSE_COL_CANDIDATES = ["CGM", "Readings (CGM / BGM)", "Readings (CGM/BGM)"]


def detect_root() -> Path | None:
    for root in ROOT_CANDIDATES:
        if root.exists() and root.is_dir():
            return root
    return None


def pick_glucose_col(cols: list[str]) -> str | None:
    for c in GLUCOSE_COL_CANDIDATES:
        if c in cols:
            return c
    return None


def load_subject_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    g_col = pick_glucose_col(df.columns.tolist())
    if g_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df["EventDateTime"], errors="coerce")
    out["glucose"] = pd.to_numeric(df[g_col], errors="coerce")
    out["carb"] = pd.to_numeric(df.get("CarbSize", 0), errors="coerce").fillna(0.0)
    out["food_delivered"] = pd.to_numeric(df.get("FoodDelivered", 0), errors="coerce").fillna(0.0)
    out["insulin"] = (
        pd.to_numeric(df.get("TotalBolusInsulinDelivered", 0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("CorrectionDelivered", 0), errors="coerce").fillna(0.0)
    )
    # Treat meal event as either FoodDelivered or CarbSize being positive.
    out["meal_event"] = (out["food_delivered"] > 0) | (out["carb"] > 0)
    out = out.dropna(subset=["datetime", "glucose"]).sort_values("datetime").reset_index(drop=True)
    out["day"] = out["datetime"].dt.date
    return out


def event_y_from_glucose(event_dt: pd.Series, day_glucose_df: pd.DataFrame) -> np.ndarray:
    if len(event_dt) == 0:
        return np.array([])
    x = day_glucose_df["datetime"].astype("int64").to_numpy()
    y = day_glucose_df["glucose"].to_numpy(dtype=float)
    x_evt = event_dt.astype("int64").to_numpy()
    x_evt = np.clip(x_evt, x.min(), x.max())
    return np.interp(x_evt, x, y)


def plot_subject_daily(df: pd.DataFrame, subject: int) -> Path:
    days = sorted(df["day"].unique())
    n_days = len(days)
    n_cols = 4
    n_rows = ceil(n_days / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 2.8 * n_rows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, day in enumerate(days):
        ax = axes[i]
        d = df[df["day"] == day].copy()
        ax.plot(d["datetime"], d["glucose"], linewidth=1.3)

        day_meals = d[d["meal_event"]]
        if not day_meals.empty:
            y_meal = event_y_from_glucose(day_meals["datetime"], d)
            ax.scatter(
                day_meals["datetime"],
                y_meal,
                s=22,
                color="#000000",
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
                label="Meal" if i == 0 else None,
            )

        day_ins = d[d["insulin"] > 0]
        if not day_ins.empty:
            y_ins = event_y_from_glucose(day_ins["datetime"], d)
            ax.scatter(
                day_ins["datetime"],
                y_ins,
                s=22,
                color="#d32f2f",
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
                label="Insulin" if i == 0 else None,
            )

        ax.set_title(str(day), fontsize=9)
        ax.set_xlabel("Time")
        ax.set_ylabel("Glucose (raw)")
        ax.grid(alpha=0.25, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(35)
            tick.set_ha("right")

    for j in range(n_days, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.suptitle(f"AZT1D Subject {subject}: Raw Glucose by Day + Meal/Insulin Events", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out = OUT_DIR / f"subject_{subject:02d}_daily_raw_glucose_with_events.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    root = detect_root()
    if root is None:
        print("No AZT1D root found. Tried:")
        for c in ROOT_CANDIDATES:
            print(f"  - {c}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_dirs = sorted([p for p in root.glob("Subject *") if p.is_dir()], key=lambda p: int(p.name.split()[-1]))
    saved = []

    for sd in subject_dirs:
        subject = int(sd.name.split()[-1])
        csv_path = sd / f"{sd.name}.csv"
        if not csv_path.exists():
            continue
        df = load_subject_df(csv_path)
        if df.empty:
            continue
        out = plot_subject_daily(df, subject)
        saved.append((subject, len(df), df["day"].nunique(), out))

    print("Saved AZT1D raw daily glucose plots:")
    for subject, n_rows, n_days, out in saved:
        print(f"  subject={subject:02d} rows={n_rows} days={n_days} -> {out}")


if __name__ == "__main__":
    main()
