import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import json
import os
from params import *
from processing_functions import *

warnings.filterwarnings('ignore')

print("Building training data")
all_patients = []

for patient in PATIENTS_AZT1D:
    file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
    df = pd.read_csv(file_path)
    df['patient'] = patient
    df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
    df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
    df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
    df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
    df['correction'] = df[AZT1D_COLUMNS['correction']].fillna(0)
    
    # Add hour and time features
    df['hour'] = df['datetime'].dt.hour
    df['time'] = df['hour'] + df['datetime'].dt.minute / 60
    
    # Keep only needed columns
    df = df[['patient', 'datetime', 'glucose', 'carbohydrates', 'insulin', 'correction', 'hour', 'time']].copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Old row-based logic (kept for reference):
    # for horizon in PREDICTION_HORIZONS:
    #     df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
    # df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
    # df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    # df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    # df = df.dropna(subset=[f'glucose_24'])
    df = build_segment_aware_glucose_features(
        df,
        glucose_col='glucose',
        datetime_col='datetime',
        horizons=PREDICTION_HORIZONS,
        history_window=HISTORY_WINDOW_POINTS,
        gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
    )
    all_patients.append(df)
features_to_remove = FEATURES_TO_REMOVE_AZT1D + PH_COLUMNS + ['patient_id']

# Load or optimize Bezier params separately for every patient
if LOAD_PARAMS:
    print("Loading existing Bezier parameters")
    patient_params = {}
    param_file = 'results/bezier_params/azt1d_all_patient_bezier_params.json'
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            all_patient_params = json.load(f)
        # Convert from "patient_X" keys back to integer keys
        for key, params in all_patient_params.items():
            p = int(key.replace('patient_', ''))
            patient_params[p] = params
        print(f"Loaded Bezier parameters for {len(patient_params)} patients")
    else:
        print(f"Parameter file {param_file} not found, will optimize new parameters")
        LOAD_PARAMS = False

if not LOAD_PARAMS:
    print("Optimizing params per patient")
    patient_params = {}
    for df in all_patients:
        p = int(df['patient'].iloc[0])
        first_14_dates = sorted(df['datetime'].dt.normalize().unique())[:14]
        g_train = df[df['datetime'].dt.normalize().isin(first_14_dates)].copy()
        c_train = g_train[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
        patient_params[p] = optimize_params(
            f'azt1d_p{p}',
            OPTIMIZATION_FEATURES_AZT1D,
            FAST_FEATURES,
            [(g_train, c_train)],
            features_to_remove,
            prediction_horizon=DEFAULT_PREDICTION_HORIZON,
            n_trials=N_TRIALS,
        )

    # Save all patient Bezier parameters in nested dict
    os.makedirs('results/bezier_params', exist_ok=True)
    all_patient_params = {f"patient_{p}": patient_params[p] for p in patient_params.keys()}
    with open('results/bezier_params/azt1d_all_patient_bezier_params.json', 'w') as f:
        json.dump(all_patient_params, f, indent=2)
    print("Saved all patient Bezier parameters to azt1d_all_patient_bezier_params.json")

results_glucose_insulin = []
results_last_meal = []
results_bezier = []

print("Training and predicting - comparing 3 approaches")
# Cache patient data (use all available days for each patient)
patient_to_data = {}
for df in all_patients:
    p = int(df['patient'].iloc[0])
    patient_data = df.copy()  # Use all available data
    comb_data = patient_data[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
    patient_to_data[p] = (patient_data, comb_data)

for prediction_horizon in PREDICTION_HORIZONS:
    target_feature = f'glucose_{prediction_horizon}'

    # Build PH-aligned datasets per patient for all 3 approaches
    frames_glucose_insulin, frames_last_meal, frames_bezier = [], [], []
    for p in patient_to_data.keys():
        g_df, c_df = patient_to_data[p]
        
        # Approach 1: Glucose + insulin only
        d1 = g_df.copy()
        # Add basic insulin features (cumulative 3h insulin)
        for idx, glucose_row in d1.iterrows():
            glucose_time = glucose_row['datetime']
            three_hours_ago = glucose_time - pd.Timedelta(hours=3)
            insulin_window = c_df[(c_df['datetime'] > three_hours_ago) & (c_df['datetime'] <= glucose_time)]
            d1.at[idx, 'cumulative_insulin_3h'] = insulin_window['insulin'].sum()
        d1['patient_id'] = f"patient_{p}"
        frames_glucose_insulin.append(d1)

        # Approach 2: Glucose + insulin + last meal information
        d2 = add_cumulative_features(g_df, c_df)
        d2['patient_id'] = f"patient_{p}"
        frames_last_meal.append(d2)

        # Approach 3: Glucose + Bezier curve features (patient-specific params)
        d3 = add_temporal_features(patient_params[p], OPTIMIZATION_FEATURES_AZT1D, g_df, c_df, prediction_horizon)
        d3['patient_id'] = f"patient_{p}"
        frames_bezier.append(d3)

    all_glucose_insulin = pd.concat(frames_glucose_insulin, ignore_index=True)
    all_last_meal = pd.concat(frames_last_meal, ignore_index=True)
    all_bezier = pd.concat(frames_bezier, ignore_index=True)

    # Evaluate all three models with stepwise retraining (like d1namo.py)
    print("Training and predicting with stepwise retraining")
    patient_ids = sorted(all_glucose_insulin['patient_id'].str.replace('patient_', '', regex=False).astype(int).unique())
    
    for p in patient_ids:
        # Get test data for all approaches
        mask1 = all_glucose_insulin['patient_id'] == f"patient_{p}"
        mask2 = all_last_meal['patient_id'] == f"patient_{p}"
        mask3 = all_bezier['patient_id'] == f"patient_{p}"
        
        all_days = sorted(all_glucose_insulin[mask1]['datetime'].dt.day.unique())
        test_days = all_days[14:]  # Test on all days after first 14
        test1 = all_glucose_insulin[mask1 & (all_glucose_insulin['datetime'].dt.day.isin(test_days))]
        test2 = all_last_meal[mask2 & (all_last_meal['datetime'].dt.day.isin(test_days))]
        test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day.isin(test_days))]
        
        for start_idx in range(0, len(test1), STEP_SIZE):
            end_idx = min(start_idx + STEP_SIZE, len(test1))
            batch1 = test1.iloc[start_idx:end_idx]
            batch2 = test2.iloc[start_idx:end_idx]
            batch3 = test3.iloc[start_idx:end_idx]
            
            if len(batch1) == 0:
                continue

            # Training data for all approaches (only this patient's data up to current batch)
            Xdf1 = all_glucose_insulin[mask1 & (all_glucose_insulin['datetime'] < batch1['datetime'].min())]
            Xdf2 = all_last_meal[mask2 & (all_last_meal['datetime'] < batch2['datetime'].min())]
            Xdf3 = all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())]

            if len(Xdf1) == 0:
                continue

            indices = train_test_split(range(len(Xdf1)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
            weights_train = [1 for _ in indices[0]]  # All same patient, so all weight 1
            weights_val = [1 for _ in indices[1]]

            # Approach 1: Glucose + insulin only
            feats1 = ['glucose', 'glucose_change', 'glucose_change_projected', 'glucose_projected', 'hour', 'time', 'cumulative_insulin_3h']
            feats1 = [f for f in feats1 if f in Xdf1.columns]
            rmse1 = train_and_predict(Xdf1, indices[0], indices[1], target_feature, feats1, batch1, weights_train, weights_val)
            results_glucose_insulin.append([prediction_horizon, p, batch1['datetime'].dt.day.iloc[0], -1, rmse1])

            # Approach 2: Glucose + insulin + last meal
            feats2 = Xdf2.columns.difference(features_to_remove)
            rmse2 = train_and_predict(Xdf2, indices[0], indices[1], target_feature, feats2, batch2, weights_train, weights_val)
            results_last_meal.append([prediction_horizon, p, batch2['datetime'].dt.day.iloc[0], -1, rmse2])

            # Approach 3: Glucose + Bezier features
            feats3 = Xdf3.columns.difference(features_to_remove)
            rmse3 = train_and_predict(Xdf3, indices[0], indices[1], target_feature, feats3, batch3, weights_train, weights_val)
            results_bezier.append([prediction_horizon, p, batch3['datetime'].dt.day.iloc[0], -1, rmse3])

    # Print horizon means
    df1 = pd.DataFrame(results_glucose_insulin, columns=['Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE'])
    df2 = pd.DataFrame(results_last_meal, columns=['Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE'])
    df3 = pd.DataFrame(results_bezier, columns=['Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE'])
    
    rmse1_ph = df1[df1['Prediction Horizon']==prediction_horizon]['RMSE'].mean()
    rmse2_ph = df2[df2['Prediction Horizon']==prediction_horizon]['RMSE'].mean()
    rmse3_ph = df3[df3['Prediction Horizon']==prediction_horizon]['RMSE'].mean()
    rmse1_std = df1[df1['Prediction Horizon']==prediction_horizon]['RMSE'].std()
    rmse2_std = df2[df2['Prediction Horizon']==prediction_horizon]['RMSE'].std()
    rmse3_std = df3[df3['Prediction Horizon']==prediction_horizon]['RMSE'].std()

    # 旧输出（只显示RMSE均值）
    # print(f"AZT1D PH {prediction_horizon}: Glucose+Insulin {rmse1_ph:.4f}, +LastMeal {rmse2_ph:.4f}, Bezier {rmse3_ph:.4f}")
    print(
        f"PH {prediction_horizon}: "
        f"Glucose+Insulin {rmse1_ph:.4f}±{rmse1_std:.4f}, "
        f"+LastMeal {rmse2_ph:.4f}±{rmse2_std:.4f}, "
        f"Bezier {rmse3_ph:.4f}±{rmse3_std:.4f}"
    )

# Combine all results into single file
all_results = []

# Add approach labels to each result set
for result in results_glucose_insulin:
    all_results.append(result + ['Glucose+Insulin'])

for result in results_last_meal:
    all_results.append(result + ['LastMeal'])

for result in results_bezier:
    all_results.append(result + ['Bezier'])

# Save combined results
pd.DataFrame(all_results, columns=['Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE', 'Approach']).to_csv('results/azt1d_comparison.csv', index=False)
print("Saved all results to azt1d_comparison.csv")
