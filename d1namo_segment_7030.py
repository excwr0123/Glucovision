import json
import os
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from params import *
from processing_functions import *

warnings.filterwarnings("ignore")


def add_cumulative_insulin_3h(g_df, c_df):
    out = g_df.copy()
    for idx, row in out.iterrows():
        t = row["datetime"]
        t0 = t - pd.Timedelta(hours=3)
        ins_window = c_df[(c_df["datetime"] > t0) & (c_df["datetime"] <= t)]
        out.at[idx, "cumulative_insulin_3h"] = ins_window["insulin"].sum() if "insulin" in ins_window.columns else 0.0
    return out


def split_patient_frame(df, ratio=0.7):
    train_df, test_df = split_train_test_by_time(df, train_ratio=ratio, datetime_col="datetime")
    if len(train_df) == 0 or len(test_df) == 0:
        return None, None
    return train_df, test_df


def build_weights(df, target_patient):
    return [CURRENT_PATIENT_WEIGHT if pid == f"patient_{target_patient}" else 1 for pid in df["patient_id"]]


features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [f"glucose_{h}" for h in PREDICTION_HORIZONS]

print("Loading D1Namo base data")
patient_to_base = {}
for p in PATIENTS_D1NAMO:
    g_base, c_base = get_d1namo_base_data(p)
    patient_to_base[p] = (g_base, c_base)

print("Preparing/Loading Bezier params (train split only, PH=12)")
patient_params = {}
for p in PATIENTS_D1NAMO:
    param_path = f"results/bezier_params/d1namo_p{p}_bezier_params.json"
    if LOAD_PARAMS and os.path.exists(param_path):
        with open(param_path, "r") as f:
            patient_params[p] = json.load(f)
        continue

    g_base, c_base = patient_to_base[p]
    g_h = build_segment_samples_for_horizon(
        g_base,
        horizon=DEFAULT_PREDICTION_HORIZON,
        history_window=HISTORY_WINDOW_POINTS,
        gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
    )
    g_train, _ = split_patient_frame(g_h, ratio=0.7)
    if g_train is None:
        print(f"[WARN] Skip param optimization for patient {p}: insufficient samples")
        continue
    c_train = c_base[c_base["datetime"] <= g_train["datetime"].max()].copy()
    patient_params[p] = optimize_params(
        f"d1namo_p{p}",
        OPTIMIZATION_FEATURES_D1NAMO,
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

    train_dict_1, test_dict_1 = {}, {}
    train_dict_2, test_dict_2 = {}, {}
    train_dict_3, test_dict_3 = {}, {}

    for p in PATIENTS_D1NAMO:
        if p not in patient_params:
            continue
        g_base, c_base = patient_to_base[p]
        g_h = build_segment_samples_for_horizon(
            g_base,
            horizon=prediction_horizon,
            history_window=HISTORY_WINDOW_POINTS,
            gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
        )
        if g_h.empty:
            continue

        # Approach 1
        d1 = add_cumulative_insulin_3h(g_h, c_base)
        d1["patient_id"] = f"patient_{p}"
        tr1, te1 = split_patient_frame(d1, ratio=0.7)

        # Approach 2
        d2 = add_cumulative_features(g_h, c_base)
        d2["patient_id"] = f"patient_{p}"
        tr2, te2 = split_patient_frame(d2, ratio=0.7)

        # Approach 3
        d3 = add_temporal_features(patient_params[p], OPTIMIZATION_FEATURES_D1NAMO, g_h, c_base, prediction_horizon)
        d3["patient_id"] = f"patient_{p}"
        tr3, te3 = split_patient_frame(d3, ratio=0.7)

        if tr1 is None or tr2 is None or tr3 is None:
            continue

        train_dict_1[p], test_dict_1[p] = tr1, te1
        train_dict_2[p], test_dict_2[p] = tr2, te2
        train_dict_3[p], test_dict_3[p] = tr3, te3

    eval_patients = sorted(set(train_dict_1.keys()) & set(train_dict_2.keys()) & set(train_dict_3.keys()))
    for p in eval_patients:
        # Approach 1
        Xdf1 = pd.concat([train_dict_1[x] for x in eval_patients], ignore_index=True)
        test1 = test_dict_1[p]
        if len(Xdf1) < 10 or len(test1) == 0:
            continue
        idx_tr1, idx_val1 = train_test_split(range(len(Xdf1)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        w_all1 = build_weights(Xdf1, p)
        w_tr1 = [w_all1[i] for i in idx_tr1]
        w_val1 = [w_all1[i] for i in idx_val1]
        feats1 = ["glucose", "glucose_change", "glucose_change_projected", "glucose_projected", "hour", "time", "cumulative_insulin_3h"]
        feats1 = [f for f in feats1 if f in Xdf1.columns]
        rmse1 = train_and_predict(Xdf1, idx_tr1, idx_val1, target_col, feats1, test1, w_tr1, w_val1)
        results.append([prediction_horizon, p, "Glucose+Insulin", rmse1, len(Xdf1), len(test1)])

        # Approach 2
        Xdf2 = pd.concat([train_dict_2[x] for x in eval_patients], ignore_index=True)
        test2 = test_dict_2[p]
        idx_tr2, idx_val2 = train_test_split(range(len(Xdf2)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        w_all2 = build_weights(Xdf2, p)
        w_tr2 = [w_all2[i] for i in idx_tr2]
        w_val2 = [w_all2[i] for i in idx_val2]
        feats2 = Xdf2.columns.difference(features_to_remove)
        rmse2 = train_and_predict(Xdf2, idx_tr2, idx_val2, target_col, feats2, test2, w_tr2, w_val2)
        results.append([prediction_horizon, p, "LastMeal", rmse2, len(Xdf2), len(test2)])

        # Approach 3
        Xdf3 = pd.concat([train_dict_3[x] for x in eval_patients], ignore_index=True)
        test3 = test_dict_3[p]
        idx_tr3, idx_val3 = train_test_split(range(len(Xdf3)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        w_all3 = build_weights(Xdf3, p)
        w_tr3 = [w_all3[i] for i in idx_tr3]
        w_val3 = [w_all3[i] for i in idx_val3]
        feats3 = Xdf3.columns.difference(features_to_remove)
        rmse3 = train_and_predict(Xdf3, idx_tr3, idx_val3, target_col, feats3, test3, w_tr3, w_val3)
        results.append([prediction_horizon, p, "Bezier", rmse3, len(Xdf3), len(test3)])

out_path = "results/d1namo_comparison_segment_7030.csv"
pd.DataFrame(
    results,
    columns=["Prediction Horizon", "Patient", "Approach", "RMSE", "Train Samples", "Test Samples"],
).to_csv(out_path, index=False)
print(f"Saved results to {out_path}")
