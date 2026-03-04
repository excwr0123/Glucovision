import json
import os
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from params import *
from processing_functions import *

warnings.filterwarnings("ignore")


GLUCOSE_COL_CANDIDATES = ["CGM", "Readings (CGM / BGM)", "Readings (CGM/BGM)"]


def pick_glucose_col(columns):
    for c in GLUCOSE_COL_CANDIDATES:
        if c in columns:
            return c
    return None


def load_azt1d_subject(patient):
    file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
    df = pd.read_csv(file_path)
    g_col = pick_glucose_col(df.columns.tolist())
    if g_col is None:
        raise ValueError(f"No glucose column found for patient {patient}")

    df["patient"] = patient
    df["datetime"] = pd.to_datetime(df[AZT1D_COLUMNS["datetime"]], errors="coerce")
    df["glucose"] = df[g_col].fillna(0)
    df["carbohydrates"] = df[AZT1D_COLUMNS["carbohydrates"]].fillna(0)
    df["insulin"] = df[AZT1D_COLUMNS["insulin"]].fillna(0)
    df["correction"] = df[AZT1D_COLUMNS["correction"]].fillna(0)
    df = df.dropna(subset=["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["time"] = df["hour"] + df["datetime"].dt.minute / 60
    return df[["patient", "datetime", "glucose", "carbohydrates", "insulin", "correction", "hour", "time"]].sort_values("datetime").reset_index(drop=True)


def add_cumulative_insulin_3h(g_df, c_df):
    out = g_df.copy()
    for idx, row in out.iterrows():
        t = row["datetime"]
        t0 = t - pd.Timedelta(hours=3)
        ins_window = c_df[(c_df["datetime"] > t0) & (c_df["datetime"] <= t)]
        out.at[idx, "cumulative_insulin_3h"] = ins_window["insulin"].sum() if "insulin" in ins_window.columns else 0.0
    return out


features_to_remove = FEATURES_TO_REMOVE_AZT1D + PH_COLUMNS + ["patient_id"]
print("Building AZT1D base data")
patient_to_base = {}
for p in PATIENTS_AZT1D:
    patient_to_base[p] = load_azt1d_subject(p)

print("Preparing/Loading Bezier params (train split only, PH=12)")
patient_params = {}
all_param_file = "results/bezier_params/azt1d_all_patient_bezier_params.json"
if LOAD_PARAMS and os.path.exists(all_param_file):
    with open(all_param_file, "r") as f:
        stored = json.load(f)
    for k, v in stored.items():
        patient_params[int(k.replace("patient_", ""))] = v

for p in PATIENTS_AZT1D:
    if p in patient_params:
        continue
    g_base = patient_to_base[p]
    g_h = build_segment_samples_for_horizon(
        g_base,
        horizon=DEFAULT_PREDICTION_HORIZON,
        history_window=HISTORY_WINDOW_POINTS,
        gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
    )
    if g_h.empty or len(g_h) < 10:
        print(f"[WARN] Skip param optimization for patient {p}: insufficient samples")
        continue
    g_train, _ = split_train_test_by_time(g_h, train_ratio=0.7, datetime_col="datetime")
    c_train = g_train[["datetime", "carbohydrates", "insulin", "correction"]].copy()
    patient_params[p] = optimize_params(
        f"azt1d_p{p}",
        OPTIMIZATION_FEATURES_AZT1D,
        FAST_FEATURES,
        [(g_train, c_train)],
        features_to_remove,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        n_trials=N_TRIALS,
    )

results = []
print("Training/evaluating with segment-aware samples + time 70/30 split")

for prediction_horizon in PREDICTION_HORIZONS:
    target_col = f"glucose_{prediction_horizon}"
    for p in PATIENTS_AZT1D:
        if p not in patient_params:
            continue
        g_base = patient_to_base[p]
        g_h = build_segment_samples_for_horizon(
            g_base,
            horizon=prediction_horizon,
            history_window=HISTORY_WINDOW_POINTS,
            gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
        )
        if len(g_h) < 10:
            continue
        c_full = g_h[["datetime", "carbohydrates", "insulin", "correction"]].copy()

        # Approach 1
        d1 = add_cumulative_insulin_3h(g_h, c_full)
        train1, test1 = split_train_test_by_time(d1, train_ratio=0.7, datetime_col="datetime")
        if len(train1) < 10 or len(test1) == 0:
            continue
        idx_tr1, idx_val1 = train_test_split(range(len(train1)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        feats1 = ["glucose", "glucose_change", "glucose_change_projected", "glucose_projected", "hour", "time", "cumulative_insulin_3h"]
        feats1 = [f for f in feats1 if f in train1.columns]
        rmse1 = train_and_predict(train1, idx_tr1, idx_val1, target_col, feats1, test1, [1] * len(idx_tr1), [1] * len(idx_val1))
        results.append([prediction_horizon, p, "Glucose+Insulin", rmse1, len(train1), len(test1)])

        # Approach 2
        d2 = add_cumulative_features(g_h, c_full)
        train2, test2 = split_train_test_by_time(d2, train_ratio=0.7, datetime_col="datetime")
        idx_tr2, idx_val2 = train_test_split(range(len(train2)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        feats2 = train2.columns.difference(features_to_remove)
        rmse2 = train_and_predict(train2, idx_tr2, idx_val2, target_col, feats2, test2, [1] * len(idx_tr2), [1] * len(idx_val2))
        results.append([prediction_horizon, p, "LastMeal", rmse2, len(train2), len(test2)])

        # Approach 3
        d3 = add_temporal_features(patient_params[p], OPTIMIZATION_FEATURES_AZT1D, g_h, c_full, prediction_horizon)
        train3, test3 = split_train_test_by_time(d3, train_ratio=0.7, datetime_col="datetime")
        idx_tr3, idx_val3 = train_test_split(range(len(train3)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        feats3 = train3.columns.difference(features_to_remove)
        rmse3 = train_and_predict(train3, idx_tr3, idx_val3, target_col, feats3, test3, [1] * len(idx_tr3), [1] * len(idx_val3))
        results.append([prediction_horizon, p, "Bezier", rmse3, len(train3), len(test3)])

out_path = "results/azt1d_comparison_segment_7030.csv"
pd.DataFrame(
    results,
    columns=["Prediction Horizon", "Patient", "Approach", "RMSE", "Train Samples", "Test Samples"],
).to_csv(out_path, index=False)
print(f"Saved results to {out_path}")
