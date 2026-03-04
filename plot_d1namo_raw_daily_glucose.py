#!/usr/bin/env python3
"""Plot raw D1Namo glucose traces per patient per day."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


PATIENTS = ["001", "002", "004", "006", "007", "008"]
DATA_ROOT = Path("diabetes_subset_pictures-glucose-food-insulin")
OUT_DIR = Path("results/plots/d1namo_raw_daily_glucose")


def load_raw_glucose(patient: str) -> pd.DataFrame:
    path = DATA_ROOT / patient / "glucose.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    # If glucose has NaN/invalid values, matplotlib will break the line.
    # Drop invalid points to keep the trace visually continuous over available data.
    df = df.dropna(subset=["datetime", "glucose"]).copy()
    df["day"] = df["datetime"].dt.date
    # Keep raw glucose values from source file (mmol/L).
    return df.sort_values("datetime").reset_index(drop=True)


def load_meal_events(patient: str) -> pd.DataFrame:
    path = DATA_ROOT / patient / "food.csv"
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "day"])
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        return pd.DataFrame(columns=["datetime", "day"])
    dt = pd.to_datetime(df["datetime"], format="%Y:%m:%d %H:%M:%S", errors="coerce")
    out = pd.DataFrame({"datetime": dt}).dropna().sort_values("datetime").reset_index(drop=True)
    out["day"] = out["datetime"].dt.date
    return out


def load_insulin_events(patient: str) -> pd.DataFrame:
    path = DATA_ROOT / patient / "insulin.csv"
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "day"])
    df = pd.read_csv(path)
    if not {"date", "time"}.issubset(df.columns):
        return pd.DataFrame(columns=["datetime", "day"])
    dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    fast = pd.to_numeric(df.get("fast_insulin", 0), errors="coerce").fillna(0)
    slow = pd.to_numeric(df.get("slow_insulin", 0), errors="coerce").fillna(0)
    dose = fast + slow
    out = pd.DataFrame({"datetime": dt, "dose": dose}).dropna(subset=["datetime"])
    out = out[out["dose"] > 0].sort_values("datetime").reset_index(drop=True)
    out["day"] = out["datetime"].dt.date
    return out


def event_y_from_glucose(event_dt: pd.Series, day_glucose_df: pd.DataFrame) -> np.ndarray:
    """Map event timestamps to glucose y-values via linear interpolation."""
    if len(event_dt) == 0:
        return np.array([])
    x = day_glucose_df["datetime"].astype("int64").to_numpy()
    y = day_glucose_df["glucose"].to_numpy(dtype=float)
    x_evt = event_dt.astype("int64").to_numpy()
    # Clamp to day edges then interpolate on existing glucose curve.
    x_evt = np.clip(x_evt, x.min(), x.max())
    return np.interp(x_evt, x, y)


def plot_patient_daily(df: pd.DataFrame, meals: pd.DataFrame, insulin: pd.DataFrame, patient: str) -> Path:
    days = sorted(df["day"].unique())
    n_days = len(days)
    n_cols = 3
    n_rows = ceil(n_days / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.0 * n_rows), sharey=True)
    if not hasattr(axes, "flatten"):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, day in enumerate(days):
        ax = axes[i]
        d = df[df["day"] == day].copy()
        ax.plot(d["datetime"], d["glucose"], linewidth=1.6)

        day_meals = meals[meals["day"] == day].copy()
        day_ins = insulin[insulin["day"] == day].copy()
        if not day_meals.empty:
            y_meal = event_y_from_glucose(day_meals["datetime"], d)
            ax.scatter(
                day_meals["datetime"],
                y_meal,
                s=26,
                color="#000000",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
                label="Meal" if i == 0 else None,
            )
        if not day_ins.empty:
            y_ins = event_y_from_glucose(day_ins["datetime"], d)
            ax.scatter(
                day_ins["datetime"],
                y_ins,
                s=26,
                color="#d32f2f",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
                label="Insulin" if i == 0 else None,
            )

        ax.set_title(str(day), fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Glucose (raw mmol/L)")
        ax.grid(alpha=0.25, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(40)
            tick.set_ha("right")

    for j in range(n_days, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.suptitle(f"D1Namo Patient {patient}: Raw Glucose by Day + Meal/Insulin Events", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = OUT_DIR / f"patient_{patient}_daily_raw_glucose_with_events.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    for patient in PATIENTS:
        df = load_raw_glucose(patient)
        meals = load_meal_events(patient)
        insulin = load_insulin_events(patient)
        out = plot_patient_daily(df, meals, insulin, patient)
        saved.append((patient, len(df), df["day"].nunique(), out))

    print("Saved D1Namo raw daily glucose plots (with meal/insulin events):")
    for patient, n_rows, n_days, out in saved:
        print(f"  patient={patient} rows={n_rows} days={n_days} -> {out}")


if __name__ == "__main__":
    main()
