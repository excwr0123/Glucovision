#!/usr/bin/env python3
"""Export one D1Namo patient's LastMeal/Bezier model inputs to CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from params import (
    FEATURES_TO_REMOVE_D1NAMO,
    OPTIMIZATION_FEATURES_D1NAMO,
    PATIENTS_D1NAMO,
    PREDICTION_HORIZONS,
)
from processing_functions import add_cumulative_features, add_temporal_features, get_d1namo_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="001", help="D1Namo patient id, e.g. 001")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon in points (default 12=60min)")
    parser.add_argument("--out-dir", default="results", help="Output directory")
    args = parser.parse_args()

    patient = args.patient
    horizon = args.horizon
    if patient not in PATIENTS_D1NAMO:
        raise ValueError(f"patient must be one of {PATIENTS_D1NAMO}, got {patient}")
    if horizon not in PREDICTION_HORIZONS:
        raise ValueError(f"horizon must be one of {PREDICTION_HORIZONS}, got {horizon}")

    g_df, c_df = get_d1namo_data(patient)
    d2 = add_cumulative_features(g_df, c_df)
    d2["patient_id"] = f"patient_{patient}"

    param_path = Path(f"results/bezier_params/d1namo_p{patient}_bezier_params.json")
    if not param_path.exists():
        raise FileNotFoundError(f"Bezier param file not found: {param_path}")
    params = json.loads(param_path.read_text())
    d3 = add_temporal_features(params, OPTIMIZATION_FEATURES_D1NAMO, g_df, c_df, prediction_horizon=horizon)
    d3["patient_id"] = f"patient_{patient}"

    features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [f"glucose_{h}" for h in PREDICTION_HORIZONS]
    target_col = f"glucose_{horizon}"

    feats_lastmeal = d2.columns.difference(features_to_remove)
    feats_bezier = d3.columns.difference(features_to_remove)

    # Export exactly what is fed to LightGBM (+ datetime/patient/target for inspection)
    out_lastmeal = d2[["datetime", "patient_id", target_col] + list(feats_lastmeal)].copy()
    out_bezier = d3[["datetime", "patient_id", target_col] + list(feats_bezier)].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"d1namo_p{patient}_ph{horizon}"
    lastmeal_path = out_dir / f"{pfx}_lastmeal_inputs.csv"
    bezier_path = out_dir / f"{pfx}_bezier_inputs.csv"

    out_lastmeal.to_csv(lastmeal_path, index=False)
    out_bezier.to_csv(bezier_path, index=False)

    print(f"Saved LastMeal inputs: {lastmeal_path} (rows={len(out_lastmeal)}, cols={len(out_lastmeal.columns)})")
    print(f"Saved Bezier inputs:   {bezier_path} (rows={len(out_bezier)}, cols={len(out_bezier.columns)})")
    print("LastMeal columns:")
    print(out_lastmeal.columns.tolist())
    print("Bezier columns:")
    print(out_bezier.columns.tolist())


if __name__ == "__main__":
    main()
