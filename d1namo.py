import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import json
import os
from params import *
from processing_functions import *

warnings.filterwarnings('ignore')

features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS]

# Cache patient data
patient_to_data = {}
for patient in PATIENTS_D1NAMO:
    g_df, c_df = get_d1namo_data(patient)
    patient_to_data[patient] = (g_df, c_df)

# Optimize Bezier params separately for every patient
print("Optimizing params per patient")
patient_params = {}
for p in PATIENTS_D1NAMO:
    g_df, c_df = patient_to_data[p]
    train_days = g_df['datetime'].dt.day.unique()[:3] #对每个病人取前三天做为参数学习数据
    g_train = g_df[g_df['datetime'].dt.day.isin(train_days)] 
    c_train = c_df[c_df['datetime'].dt.day.isin(train_days)]
    patient_params[p] = optimize_params(
        f'd1namo_p{p}',
        OPTIMIZATION_FEATURES_D1NAMO,
        FAST_FEATURES,
        [(g_train, c_train)],
        features_to_remove,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        n_trials=N_TRIALS,
    )

# Save all patient Bezier parameters in nested dict
os.makedirs('results/bezier_params', exist_ok=True)
all_patient_params = {f"patient_{p}": patient_params[p] for p in PATIENTS_D1NAMO}
with open('results/bezier_params/d1namo_all_patient_bezier_params.json', 'w') as f:
    json.dump(all_patient_params, f, indent=2)
print("Saved all patient Bezier parameters to d1namo_all_patient_bezier_params.json")
results_glucose_insulin = []
results_last_meal = []
results_bezier = []

print("Training and predicting - comparing 3 approaches")

for prediction_horizon in PREDICTION_HORIZONS:
    target_feature = f'glucose_{prediction_horizon}'

    # Build PH-aligned datasets per patient for all 3 approaches
    frames_glucose_insulin, frames_last_meal, frames_bezier = [], [], []
    for p in PATIENTS_D1NAMO:
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

        # Approach 3: Glucose + Bezier curve features
        d3 = add_temporal_features(patient_params[p], OPTIMIZATION_FEATURES_D1NAMO, g_df, c_df, prediction_horizon)
        d3['patient_id'] = f"patient_{p}"
        frames_bezier.append(d3)

    all_glucose_insulin = pd.concat(frames_glucose_insulin, ignore_index=True)
    all_last_meal = pd.concat(frames_last_meal, ignore_index=True)
    all_bezier = pd.concat(frames_bezier, ignore_index=True)

    # Evaluate all three models
    for patient in PATIENTS_D1NAMO:
        # Get test data for all approaches
        mask1 = all_glucose_insulin['patient_id'] == f"patient_{patient}"
        mask2 = all_last_meal['patient_id'] == f"patient_{patient}"
        mask3 = all_bezier['patient_id'] == f"patient_{patient}"
        
        test_days = all_glucose_insulin[mask1]['datetime'].dt.day.unique()
        test1 = all_glucose_insulin[mask1 & (all_glucose_insulin['datetime'].dt.day >= test_days[3])]
        test2 = all_last_meal[mask2 & (all_last_meal['datetime'].dt.day >= test_days[3])]
        test3 = all_bezier[mask3 & (all_bezier['datetime'].dt.day >= test_days[3])]

        for start_idx in range(0, len(test1), STEP_SIZE):
            end_idx = min(start_idx + STEP_SIZE, len(test1))
            batch1 = test1.iloc[start_idx:end_idx]
            batch2 = test2.iloc[start_idx:end_idx]
            batch3 = test3.iloc[start_idx:end_idx]
            
            if len(batch1) == 0:
                continue

            # Training data for all approaches
            Xdf1 = pd.concat([
                all_glucose_insulin[mask1 & (all_glucose_insulin['datetime'] < batch1['datetime'].min())],
                all_glucose_insulin[~mask1],
            ])
            Xdf2 = pd.concat([
                all_last_meal[mask2 & (all_last_meal['datetime'] < batch2['datetime'].min())],
                all_last_meal[~mask2],
            ])
            Xdf3 = pd.concat([
                all_bezier[mask3 & (all_bezier['datetime'] < batch3['datetime'].min())],
                all_bezier[~mask3],
            ])

            indices = train_test_split(range(len(Xdf1)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
            #目标病人权重10，其他病人权重1
            weights_train = [CURRENT_PATIENT_WEIGHT if Xdf1['patient_id'].iloc[idx] == f"patient_{patient}" else 1 for idx in indices[0]]
            weights_val = [CURRENT_PATIENT_WEIGHT if Xdf1['patient_id'].iloc[idx] == f"patient_{patient}" else 1 for idx in indices[1]]

            # Approach 1: Glucose + insulin only
            feats1 = ['glucose', 'glucose_change', 'glucose_change_projected', 'glucose_projected', 'hour', 'time', 'cumulative_insulin_3h']
            feats1 = [f for f in feats1 if f in Xdf1.columns]
            rmse1 = train_and_predict(Xdf1, indices[0], indices[1], target_feature, feats1, batch1, weights_train, weights_val)
            results_glucose_insulin.append([prediction_horizon, patient, batch1['datetime'].dt.day.iloc[0], -1, rmse1])

            # Approach 2: Glucose + insulin + last meal
            feats2 = Xdf2.columns.difference(features_to_remove)
            rmse2 = train_and_predict(Xdf2, indices[0], indices[1], target_feature, feats2, batch2, weights_train, weights_val)
            results_last_meal.append([prediction_horizon, patient, batch2['datetime'].dt.day.iloc[0], -1, rmse2])

            # Approach 3: Glucose + Bezier features
            feats3 = Xdf3.columns.difference(features_to_remove)
            rmse3 = train_and_predict(Xdf3, indices[0], indices[1], target_feature, feats3, batch3, weights_train, weights_val)
            results_bezier.append([prediction_horizon, patient, batch3['datetime'].dt.day.iloc[0], -1, rmse3])

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
    # print(f"PH {prediction_horizon}: Glucose+Insulin {rmse1_ph:.4f}, +LastMeal {rmse2_ph:.4f}, Bezier {rmse3_ph:.4f}")
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
pd.DataFrame(all_results, columns=['Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE', 'Approach']).to_csv('results/d1namo_comparison.csv', index=False)
print("Saved all results to d1namo_comparison.csv")
