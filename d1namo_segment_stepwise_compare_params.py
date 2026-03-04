#!/usr/bin/env python3
"""Compare D1Namo segment-aware + stepwise retrain under two Bezier param modes.

Modes:
1) load_existing: use existing d1namo_p{patient}_bezier_params.json
2) reoptimize: force re-optimization (LOAD_PARAMS=False) and save to separate files
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import processing_functions as pf
from params import (
    CURRENT_PATIENT_WEIGHT,
    DEFAULT_PREDICTION_HORIZON,
    FEATURES_TO_REMOVE_D1NAMO,
    FAST_FEATURES,
    HISTORY_WINDOW_POINTS,
    N_TRIALS,
    PATIENTS_D1NAMO,
    PREDICTION_HORIZONS,
    RANDOM_SEED,
    SEGMENT_GAP_THRESHOLD_MINUTES,
    STEP_SIZE,
    VALIDATION_SIZE,
    OPTIMIZATION_FEATURES_D1NAMO,
)

warnings.filterwarnings("ignore")


def add_cumulative_insulin_3h(g_df: pd.DataFrame, c_df: pd.DataFrame) -> pd.DataFrame:
    out = g_df.copy()
    for idx, row in out.iterrows():
        glucose_time = row["datetime"]
        three_hours_ago = glucose_time - pd.Timedelta(hours=3)
        insulin_window = c_df[(c_df["datetime"] > three_hours_ago) & (c_df["datetime"] <= glucose_time)]
        out.at[idx, "cumulative_insulin_3h"] = insulin_window["insulin"].sum() if "insulin" in insulin_window.columns else 0.0
    return out


def rmse_pm_std_table(df: pd.DataFrame, mode_col: str = "Mode") -> pd.DataFrame:
    grp = (
        df.groupby([mode_col, "Prediction Horizon", "Approach"], as_index=False)
        .agg(rmse_mean=("RMSE", "mean"), rmse_std=("RMSE", "std"), n=("RMSE", "count"))
        .sort_values([mode_col, "Prediction Horizon", "Approach"])
    )
    grp["RMSE±Std"] = grp["rmse_mean"].map(lambda x: f"{x:.3f}") + "±" + grp["rmse_std"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "nan"
    )
    return grp


def run_experiment(mode: str, n_trials: int, strict_existing: bool) -> pd.DataFrame:
    assert mode in {"load_existing", "reoptimize"}
    pf.LOAD_PARAMS = mode == "load_existing"
    features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [f"glucose_{h}" for h in PREDICTION_HORIZONS]

    print(f"\n=== Running mode: {mode} ===")
    print(f"segment_gap>{SEGMENT_GAP_THRESHOLD_MINUTES}min, history={HISTORY_WINDOW_POINTS}, step={STEP_SIZE}")
    print(f"LOAD_PARAMS={pf.LOAD_PARAMS}, n_trials={n_trials}")

    # Load base data once
    patient_to_base = {}
    for p in PATIENTS_D1NAMO:
        g_base, c_base = pf.get_d1namo_base_data(p)
        patient_to_base[p] = (g_base, c_base)

    # Prepare params per patient
    patient_params = {}
    for p in PATIENTS_D1NAMO:
        g_base, c_base = patient_to_base[p]
        # Keep "load_existing" mode tied to original param files.
        if mode == "load_existing":
            param_file = Path(f"results/bezier_params/d1namo_p{p}_bezier_params.json")
            if not param_file.exists() and strict_existing:
                raise FileNotFoundError(f"Missing existing param file: {param_file}")
            approach_name = f"d1namo_p{p}"
        else:
            # Save re-optimized params separately to avoid overwriting old files.
            approach_name = f"d1namo_seg_stepwise_reopt_p{p}"

        # Use the same training window strategy as original script for param tuning:
        # first 3 calendar days from segment-aware samples at default horizon.
        g_default = pf.build_segment_samples_for_horizon(
            g_base,
            horizon=DEFAULT_PREDICTION_HORIZON,
            history_window=HISTORY_WINDOW_POINTS,
            gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
        )
        if g_default.empty:
            print(f"[WARN] Patient {p}: no samples for default horizon, skipping")
            continue
        train_days = g_default["datetime"].dt.day.unique()[:3]
        g_train = g_default[g_default["datetime"].dt.day.isin(train_days)].copy()
        if g_train.empty:
            print(f"[WARN] Patient {p}: no training rows in first 3 days after segmentation, skipping")
            continue
        c_train = c_base[c_base["datetime"].dt.day.isin(train_days)].copy()

        patient_params[p] = pf.optimize_params(
            approach_name,
            OPTIMIZATION_FEATURES_D1NAMO,
            FAST_FEATURES,
            [(g_train, c_train)],
            features_to_remove,
            prediction_horizon=DEFAULT_PREDICTION_HORIZON,
            n_trials=n_trials,
        )

    results = []

    for prediction_horizon in PREDICTION_HORIZONS:
        target_feature = f"glucose_{prediction_horizon}"
        frames_glucose_insulin, frames_last_meal, frames_bezier = [], [], []

        for p in PATIENTS_D1NAMO:
            if p not in patient_params:
                continue
            g_base, c_base = patient_to_base[p]
            g_h = pf.build_segment_samples_for_horizon(
                g_base,
                horizon=prediction_horizon,
                history_window=HISTORY_WINDOW_POINTS,
                gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
            )
            if g_h.empty:
                continue

            d1 = add_cumulative_insulin_3h(g_h, c_base)
            d1["patient_id"] = f"patient_{p}"
            frames_glucose_insulin.append(d1)

            d2 = pf.add_cumulative_features(g_h, c_base)
            d2["patient_id"] = f"patient_{p}"
            frames_last_meal.append(d2)

            d3 = pf.add_temporal_features(patient_params[p], OPTIMIZATION_FEATURES_D1NAMO, g_h, c_base, prediction_horizon)
            d3["patient_id"] = f"patient_{p}"
            frames_bezier.append(d3)

        if not frames_glucose_insulin or not frames_last_meal or not frames_bezier:
            continue

        all_glucose_insulin = pd.concat(frames_glucose_insulin, ignore_index=True)
        all_last_meal = pd.concat(frames_last_meal, ignore_index=True)
        all_bezier = pd.concat(frames_bezier, ignore_index=True)

        for patient in PATIENTS_D1NAMO:
            mask1 = all_glucose_insulin["patient_id"] == f"patient_{patient}"
            mask2 = all_last_meal["patient_id"] == f"patient_{patient}"
            mask3 = all_bezier["patient_id"] == f"patient_{patient}"
            if not mask1.any() or not mask2.any() or not mask3.any():
                continue

            # Same day-based testing policy as original d1namo.py
            test_days = all_glucose_insulin[mask1]["datetime"].dt.day.unique()
            if len(test_days) < 4:
                continue
            test1 = all_glucose_insulin[mask1 & (all_glucose_insulin["datetime"].dt.day >= test_days[3])]
            test2 = all_last_meal[mask2 & (all_last_meal["datetime"].dt.day >= test_days[3])]
            test3 = all_bezier[mask3 & (all_bezier["datetime"].dt.day >= test_days[3])]
            if len(test1) == 0:
                continue

            for start_idx in range(0, len(test1), STEP_SIZE):
                end_idx = min(start_idx + STEP_SIZE, len(test1))
                batch1 = test1.iloc[start_idx:end_idx]
                batch2 = test2.iloc[start_idx:end_idx]
                batch3 = test3.iloc[start_idx:end_idx]
                if len(batch1) == 0:
                    continue

                Xdf1 = pd.concat(
                    [
                        all_glucose_insulin[mask1 & (all_glucose_insulin["datetime"] < batch1["datetime"].min())],
                        all_glucose_insulin[~mask1],
                    ]
                )
                Xdf2 = pd.concat(
                    [
                        all_last_meal[mask2 & (all_last_meal["datetime"] < batch2["datetime"].min())],
                        all_last_meal[~mask2],
                    ]
                )
                Xdf3 = pd.concat(
                    [
                        all_bezier[mask3 & (all_bezier["datetime"] < batch3["datetime"].min())],
                        all_bezier[~mask3],
                    ]
                )
                if len(Xdf1) < 10 or len(Xdf2) < 10 or len(Xdf3) < 10:
                    continue

                indices = train_test_split(range(len(Xdf1)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
                weights_train = [
                    CURRENT_PATIENT_WEIGHT if Xdf1["patient_id"].iloc[idx] == f"patient_{patient}" else 1
                    for idx in indices[0]
                ]
                weights_val = [
                    CURRENT_PATIENT_WEIGHT if Xdf1["patient_id"].iloc[idx] == f"patient_{patient}" else 1
                    for idx in indices[1]
                ]

                feats1 = [
                    "glucose",
                    "glucose_change",
                    "glucose_change_projected",
                    "glucose_projected",
                    "hour",
                    "time",
                    "cumulative_insulin_3h",
                ]
                feats1 = [f for f in feats1 if f in Xdf1.columns]
                rmse1 = pf.train_and_predict(Xdf1, indices[0], indices[1], target_feature, feats1, batch1, weights_train, weights_val)
                results.append([mode, prediction_horizon, patient, "Glucose+Insulin", rmse1])

                feats2 = Xdf2.columns.difference(features_to_remove)
                rmse2 = pf.train_and_predict(Xdf2, indices[0], indices[1], target_feature, feats2, batch2, weights_train, weights_val)
                results.append([mode, prediction_horizon, patient, "LastMeal", rmse2])

                feats3 = Xdf3.columns.difference(features_to_remove)
                rmse3 = pf.train_and_predict(Xdf3, indices[0], indices[1], target_feature, feats3, batch3, weights_train, weights_val)
                results.append([mode, prediction_horizon, patient, "Bezier", rmse3])

    return pd.DataFrame(results, columns=["Mode", "Prediction Horizon", "Patient", "Approach", "RMSE"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help="Optuna trials for (re)optimization mode.")
    parser.add_argument(
        "--strict-existing",
        action="store_true",
        help="In load_existing mode, fail if d1namo_p*.json is missing.",
    )
    parser.add_argument("--skip-load-existing", action="store_true", help="Skip load_existing run.")
    parser.add_argument("--skip-reoptimize", action="store_true", help="Skip reoptimize run.")
    parser.add_argument("--out-prefix", default="results/d1namo_segment_stepwise", help="Output prefix path.")
    args = parser.parse_args()

    outputs = []
    if not args.skip_load_existing:
        df_load = run_experiment("load_existing", n_trials=args.n_trials, strict_existing=args.strict_existing)
        out_load = f"{args.out_prefix}_load_existing.csv"
        df_load.to_csv(out_load, index=False)
        print(f"Saved: {out_load}")
        outputs.append(df_load)

    if not args.skip_reoptimize:
        df_reopt = run_experiment("reoptimize", n_trials=args.n_trials, strict_existing=False)
        out_reopt = f"{args.out_prefix}_reoptimize.csv"
        df_reopt.to_csv(out_reopt, index=False)
        print(f"Saved: {out_reopt}")
        outputs.append(df_reopt)

    if not outputs:
        print("No mode selected. Nothing to run.")
        return

    all_df = pd.concat(outputs, ignore_index=True)
    out_all = f"{args.out_prefix}_all_runs.csv"
    all_df.to_csv(out_all, index=False)
    print(f"Saved: {out_all}")

    stats = rmse_pm_std_table(all_df, mode_col="Mode")
    out_stats = f"{args.out_prefix}_rmse_std_comparison.csv"
    stats.to_csv(out_stats, index=False)
    print(f"Saved: {out_stats}")

    piv = stats.pivot(index=["Prediction Horizon", "Approach"], columns="Mode", values="RMSE±Std")
    print("\n=== RMSE±Std Comparison (segment-aware + stepwise) ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
