import pandas as pd
import numpy as np
import os
from scipy.special import comb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import json
from params import *

_GPU_FALLBACK_WARNED = False

def add_cumulative_features(glucose_data, combined_data):
    """Add cumulative features efficiently: time since last meal, last meal macros, cumulative insulin (2h)

    This implementation avoids per-row Python loops by using merge_asof joins and
    cumulative sums, reducing complexity from O(N*M) to roughly O(N log M).
    """
    # Ensure sorted inputs
    g = glucose_data.sort_values('datetime').reset_index() # 血糖表按时间排序，并保留原始索引
    c = combined_data.sort_values('datetime').reset_index(drop=True) # 事件表按时间排序，重置索引 （吃饭事件+胰岛素事件 按时间混在一起）
    #按照时间处理 是因为 后面要找 最近之前的一餐事件 和 最近之前的胰岛素事件，必须保证时间顺序正确

    # Identify meal events (any row with non-zero macronutrients)
    macro_cols = [
        col for col in ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'carbohydrates']
        if col in c.columns
    ]
    """python - <<'PY' 监测输出
        from processing_functions import get_d1namo_data, add_cumulative_features

        g_df, c_df = get_d1namo_data('001')
        _ = add_cumulative_features(g_df, c_df)
        PY"""

    # print(f"macro_cols={macro_cols}")
    # print(c.head())

    # Last meal merge (backward asof)
    if macro_cols: #对每个血糖时间点，找“它之前最近的一餐”
        meals = c.loc[c[macro_cols].sum(axis=1) > 0, ['datetime'] + macro_cols].copy()  #只保留吃饭事件
        meals = meals.rename(columns={'datetime': 'last_meal_time'})
        asof_meal = pd.merge_asof(
            g[['index', 'datetime']].sort_values('datetime'),
            meals.sort_values('last_meal_time'),
            left_on='datetime',
            right_on='last_meal_time',
            direction='backward' #最近一次已经吃饭的事件 保证不是未来  -> 我现在这个血糖时间点，往回看，最近一条吃饭记录是谁
        )

        # print(meals.head(10))
        #print(asof_meal.head(25))

        # Time since last meal (hours)     #改了代码 这里
        # 旧写法（错误-> 因为输出都是0）：
        # time_since = (asof_meal['datetime'] - asof_meal['datetime'].where(asof_meal[macro_cols].notna().any(axis=1)))
        # time_since_hours = time_since.dt.total_seconds().div(3600)
        time_since_hours = (asof_meal['datetime'] - asof_meal['last_meal_time']).dt.total_seconds().div(3600)
        g['time_since_last_meal'] = time_since_hours.fillna(24.0)
        #print(g[['datetime', 'time_since_last_meal']].head(25))
        
        # Last meal macros
        for col in macro_cols:
            g[f'last_meal_{col}'] = asof_meal[col].fillna(0.0)
    else:
        # If no macro columns exist, still provide defaults
        g['time_since_last_meal'] = 24.0

    # Cumulative insulin over last 2 hours via cum-sum differences with asof
    if 'insulin' in c.columns:
        ins = c[['datetime', 'insulin']].copy()
        ins['ins_cum'] = ins['insulin'].cumsum()

        # Value at t
        at_t = pd.merge_asof(
            g[['datetime']].rename(columns={'datetime': 'dt'}).sort_values('dt'),
            ins[['datetime', 'ins_cum']].rename(columns={'datetime': 'dt'}).sort_values('dt'),
            on='dt',
            direction='backward'
        )['ins_cum'].fillna(0.0)

        # Value at t-2h
        t_minus = g[['datetime']].copy()
        t_minus['dt'] = t_minus['datetime'] - pd.Timedelta(hours=2)
        at_tm2h = pd.merge_asof(
            t_minus[['dt']].sort_values('dt'),
            ins[['datetime', 'ins_cum']].rename(columns={'datetime': 'dt'}).sort_values('dt'),
            on='dt',
            direction='backward'
        )['ins_cum'].fillna(0.0)

        g['cumulative_insulin_2h'] = (at_t - at_tm2h).values

    # Restore original order and assign back into glucose_data
    out = glucose_data.copy()
    out.loc[g['index'], [col for col in g.columns if col not in ['index']]] = g.drop(columns=['index'])
    return out

def bezier_curve(points, num=50): #points：控制点（决定曲线形状） num=50：最终曲线上取 50 个点
    """Generate Bezier curve from control points using Bernstein polynomials""" # optimize_params() points优化来的
    points = np.array(points).reshape(-1, 2)
    points[0] = [0.0, 0.0]
    control_points = points[1:].copy()
    sorted_indices = np.argsort(control_points[:, 0]) #按 x 从小到大排序，得到“排序后的索引”
    points[1:] = control_points[sorted_indices]
    points[-1, 1] = 0.0
    n = len(points) - 1  #Bezier 的阶数,如果 4 个点，n=3（三次Bezier）
    t = np.linspace(0, 1, num) #在 0 到 1 之间等间距取 num 个参数点
    curve = np.zeros((num, 2))
    for i, point in enumerate(points):
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), point) #Bernstein 基函数
    return curve[np.argsort(curve[:, 0])]

def get_projected_value(window, prediction_horizon):
    """Project future value using polynomial regression"""
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window, deg=3)
    return np.polyval(coeffs, len(window) + prediction_horizon)


def build_segment_aware_glucose_features(
    glucose_df,
    glucose_col="glucose",
    datetime_col="datetime",
    horizons=None,
    history_window=HISTORY_WINDOW_POINTS,
    gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
):
    """Build labels/features within continuous segments only.

    Segment boundary rule:
      - start of series
      - any time gap > gap_threshold_min
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS

    df = glucose_df.sort_values(datetime_col).reset_index(drop=True).copy()
    if df.empty:
        return df

    delta_min = df[datetime_col].diff().dt.total_seconds().div(60)
    df["_segment_id"] = (delta_min.isna() | (delta_min > gap_threshold_min)).cumsum().astype(int)

    max_h = max(horizons)
    min_len_needed = history_window + max_h
    out_frames = []

    for _, seg in df.groupby("_segment_id", sort=False):
        seg = seg.copy()
        if len(seg) < min_len_needed:
            continue

        for horizon in horizons:
            seg[f"glucose_{horizon}"] = seg[glucose_col].shift(-horizon) - seg[glucose_col]

        seg["glucose_change"] = seg[glucose_col] - seg[glucose_col].shift(1)
        seg["glucose_change_projected"] = seg["glucose_change"].rolling(history_window, min_periods=history_window).apply(
            lambda window: get_projected_value(window, history_window)
        )
        seg["glucose_projected"] = seg[glucose_col].rolling(history_window, min_periods=history_window).apply(
            lambda window: get_projected_value(window, history_window)
        )

        seg = seg.dropna(
            subset=[f"glucose_{max_h}", "glucose_change", "glucose_change_projected", "glucose_projected"]
        )
        if not seg.empty:
            out_frames.append(seg)

    if not out_frames:
        return df.iloc[0:0].copy()
    return pd.concat(out_frames, ignore_index=True)


def build_segment_samples_for_horizon(
    glucose_df,
    horizon,
    glucose_col="glucose",
    datetime_col="datetime",
    history_window=HISTORY_WINDOW_POINTS,
    gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
):
    """Build horizon-specific samples within continuous segments.

    Per-segment usable samples follow:
      samples = max(0, segment_len - (history_window + horizon) + 1)
    """
    df = glucose_df.sort_values(datetime_col).reset_index(drop=True).copy()
    if df.empty:
        return df

    delta_min = df[datetime_col].diff().dt.total_seconds().div(60)
    df["_segment_id"] = (delta_min.isna() | (delta_min > gap_threshold_min)).cumsum().astype(int)
    out_frames = []

    for _, seg in df.groupby("_segment_id", sort=False):
        seg = seg.copy()
        if len(seg) < (history_window + horizon):
            continue

        target_col = f"glucose_{horizon}"
        seg[target_col] = seg[glucose_col].shift(-horizon) - seg[glucose_col]
        seg["glucose_change"] = seg[glucose_col] - seg[glucose_col].shift(1)
        seg["glucose_change_projected"] = seg["glucose_change"].rolling(history_window, min_periods=history_window).apply(
            lambda window: get_projected_value(window, history_window)
        )
        seg["glucose_projected"] = seg[glucose_col].rolling(history_window, min_periods=history_window).apply(
            lambda window: get_projected_value(window, history_window)
        )

        seg = seg.dropna(subset=[target_col, "glucose_change", "glucose_change_projected", "glucose_projected"])
        if not seg.empty:
            out_frames.append(seg)

    if not out_frames:
        return df.iloc[0:0].copy()
    return pd.concat(out_frames, ignore_index=True)


def split_train_test_by_time(df, train_ratio=0.7, datetime_col="datetime"):
    """Time-ordered split for one patient's dataframe."""
    if df.empty:
        return df.copy(), df.copy()
    s = df.sort_values(datetime_col).reset_index(drop=True)
    split_idx = int(len(s) * train_ratio)
    split_idx = min(max(split_idx, 1), len(s) - 1) if len(s) > 1 else len(s)
    return s.iloc[:split_idx].copy(), s.iloc[split_idx:].copy()


def get_d1namo_base_data(patient):
    """Load D1namo base data without horizon labels."""
    glucose_data = pd.read_csv(f"{D1NAMO_DATA_PATH}/{patient}/glucose.csv")
    insulin_data = pd.read_csv(f"{D1NAMO_DATA_PATH}/{patient}/insulin.csv")
    food_data = pd.read_csv(f"{FOOD_DATA_PATH}/{patient}.csv")

    glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + " " + glucose_data["time"])
    glucose_data = glucose_data.drop(["type", "comments", "date", "time"], axis=1)
    glucose_data["glucose"] *= GLUCOSE_CONVERSION_FACTOR
    glucose_data["hour"] = glucose_data["datetime"].dt.hour
    glucose_data["time"] = glucose_data["hour"] + glucose_data["datetime"].dt.minute / 60

    insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + " " + insulin_data["time"])
    insulin_data.fillna(0, inplace=True)
    insulin_data["insulin"] = insulin_data["slow_insulin"] + insulin_data["fast_insulin"]
    insulin_data = insulin_data.drop(["slow_insulin", "fast_insulin", "comment", "date", "time"], axis=1)

    food_data["datetime"] = pd.to_datetime(food_data["datetime"], format="%Y:%m:%d %H:%M:%S")
    food_data = food_data[["datetime", "simple_sugars", "complex_sugars", "proteins", "fats", "dietary_fibers"]]

    combined_data = pd.concat([food_data, insulin_data]).sort_values("datetime").reset_index(drop=True)
    combined_data.fillna(0, inplace=True)
    return glucose_data.sort_values("datetime").reset_index(drop=True), combined_data

def get_d1namo_data(patient):
    """Load D1namo data for a patient"""
    glucose_data, combined_data = get_d1namo_base_data(patient)
    
    # Old row-based logic (kept for reference):
    # for horizon in PREDICTION_HORIZONS:
    #     glucose_data[f'glucose_{horizon}'] = glucose_data['glucose'].shift(-horizon) - glucose_data['glucose']
    # glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
    # glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    # glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    # glucose_data.dropna(subset=[f'glucose_24'], inplace=True)
    glucose_data = build_segment_aware_glucose_features(
        glucose_data,
        glucose_col="glucose",
        datetime_col="datetime",
        horizons=PREDICTION_HORIZONS,
        history_window=HISTORY_WINDOW_POINTS,
        gap_threshold_min=SEGMENT_GAP_THRESHOLD_MINUTES,
    )
    glucose_data['patient_id'] = patient

    # print(f"glucose_data:")
    # print(glucose_data.head())
    # print(f"combined_data:")
    # print(combined_data.head())

    return glucose_data, combined_data

def add_temporal_features(params, features, glucose_data, combined_data, prediction_horizon): #在d1namo.py 的line74 调用
    """Add temporal features using Bezier curves (dataset-agnostic) with batched computation.
    features：要处理的特征名列表（D1namo里是6个：2个糖/蛋白/脂肪/纤维/胰岛素）
    add_temporal_features 不是找“上一顿饭”，而是把“很多历史事件”按时间衰减/变化曲线加权后，合成为当前时刻的动态输入特征
    Computes mapping in batches to avoid building a full (len(glucose) x len(combined))
    time-difference matrix, which can be very large. Also skips event rows where the
    source feature is zero to reduce work.
    """
    result = glucose_data.copy()
    g_times = result['datetime'].values.astype('datetime64[ns]').astype(np.int64)

    for feature in features:
        # Skip if source column not present
        if feature not in combined_data.columns:
            result[feature] = 0.0
            continue

        # Pre-filter combined data rows that matter for this feature
        src = combined_data[['datetime', feature]].copy()
        src = src[src[feature] != 0]
        if src.empty:
            result[feature] = 0.0
            continue

        src_times = src['datetime'].values.astype('datetime64[ns]').astype(np.int64)
        src_vals = src[feature].values.astype(float)

        curve = bezier_curve(np.array(params[feature]).reshape(-1, 2), num=32)
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        max_h = float(x_curve[-1])

        mapped = np.zeros(len(result), dtype=float) #准备每个血糖时刻的最终特征值
        # Process glucose timestamps in batches to limit memory
        batch_size = 2048
        for start in range(0, len(result), batch_size):
            end = min(start + batch_size, len(result))
            gt_batch = g_times[start:end]
            # Compute time differences (hours) for the batch
            td_hours = (gt_batch[:, None] - src_times[None, :]) / 3.6e12  #得到一个矩阵：[血糖点数 x 事件数]
            # Mask valid window [0, max_h]
            valid = (td_hours >= 0.0) & (td_hours <= max_h) #只保留“事件发生在过去，且还在有效影响窗口内”的组合
            if not valid.any():
                continue
            # Indices along curve for valid diffs
            idx = np.searchsorted(x_curve, td_hours[valid], side='left')
            idx = np.clip(idx, 0, len(y_curve) - 1)
            # Build weights matrix sparsely via zeros then fill valid positions
            weights = np.zeros_like(td_hours)
            weights[valid] = y_curve[idx]
            # Weighted sum over source events
            mapped[start:end] = weights.dot(src_vals)

        # Old row-based shift (kept for reference):
        # result[feature] = pd.Series(mapped, index=result.index).shift(-prediction_horizon)
        # Segment-aware shift to avoid crossing discontinuous gaps.
        mapped_series = pd.Series(mapped, index=result.index)
        if "_segment_id" in result.columns:
            result[feature] = mapped_series.groupby(result["_segment_id"], sort=False).shift(-prediction_horizon)
        else:
            result[feature] = mapped_series.shift(-prediction_horizon)
        # result[f'{feature}_no_shift'] = pd.Series(mapped, index=result.index)
        # result[feature] = result[f'{feature}_no_shift'].shift(-prediction_horizon)

        """
        打印输出，检查结果
            python - <<'PY'
            import json
            from processing_functions import get_d1namo_data, add_temporal_features
            from params import OPTIMIZATION_FEATURES_D1NAMO

            p = '001'
            g_df, c_df = get_d1namo_data(p)
            params = json.load(open(f'results/bezier_params/d1namo_p{p}_bezier_params.json'))

            result = add_temporal_features(params, OPTIMIZATION_FEATURES_D1NAMO, g_df, c_df, prediction_horizon=12)

            print("shape:", result.shape)
            print("columns:", result.columns.tolist())
            print(result[['datetime','glucose','simple_sugars','complex_sugars','proteins','fats','dietary_fibers','insulin']].head(20).to_string(index=False))
            PY
        """

    return result

def modify_time(glucose_data, target_hour):  #函数定义了 但是没有被调用过
    """Modify the time of day for all glucose data points while preserving date."""
    modified_data = glucose_data.copy()
    original_dates = modified_data['datetime'].dt.date
    original_minutes = modified_data['datetime'].dt.minute
    modified_data['datetime'] = pd.to_datetime([
        f"{date} {target_hour:02d}:{minute:02d}:00" 
        for date, minute in zip(original_dates, original_minutes)
    ])
    modified_data['hour'] = target_hour
    modified_data['time'] = target_hour + original_minutes / 60
    return modified_data

def train_and_predict(Xdf, idx_train, idx_val, target_col, feature_cols, test_batch, weights_train=None, weights_val=None, use_monotone=True): #use_monotone：是否启用单调约束
    lgb_params = LGB_PARAMS.copy()
    if use_monotone:
        lgb_params['monotone_constraints'] = [MONOTONE_MAP.get(col, 0) for col in feature_cols]
    X_train = Xdf[feature_cols].iloc[idx_train].values
    y_train = Xdf[target_col].iloc[idx_train].values
    X_val = Xdf[feature_cols].iloc[idx_val].values
    y_val = Xdf[target_col].iloc[idx_val].values
    train_ds = lgb.Dataset(X_train, label=y_train, weight=weights_train)
    val_ds = lgb.Dataset(X_val, label=y_val, weight=weights_val)
    try:
        model = lgb.train(lgb_params, train_ds, valid_sets=[val_ds])
    except Exception as e:
        # Robust fallback for environments where GPU exists but LightGBM isn't built with GPU support.
        if lgb_params.get('device') == 'gpu':
            global _GPU_FALLBACK_WARNED
            if not _GPU_FALLBACK_WARNED:
                print(f"[WARN] LightGBM GPU unavailable, falling back to CPU. Reason: {e}")
                _GPU_FALLBACK_WARNED = True
            lgb_params_cpu = lgb_params.copy()
            lgb_params_cpu.pop('device', None)
            model = lgb.train(lgb_params_cpu, train_ds, valid_sets=[val_ds])
        else:
            raise
    preds = model.predict(test_batch[feature_cols].values)
    return float(np.sqrt(mean_squared_error(test_batch[target_col].values, preds)))

def optimize_params(
    approach_name,
    features,
    fast_features,
    train_data,
    features_to_remove,
    prediction_horizon=12,
    n_trials=N_TRIALS,  #n_trials：Optuna尝试次数
):  #optimize_params 是“给 Bezier 找最佳控制点参数”的函数 让 add_temporal_features 生成的特征，喂给 LightGBM 后 RMSE 最小
    if LOAD_PARAMS:
        path = f"results/bezier_params/{approach_name}_bezier_params.json"
        if os.path.exists(path):
            return json.load(open(path))

    max_x_values = np.where(np.isin(features, fast_features), MAX_X_VALUES_FAST, MAX_X_VALUES_SLOW)

    def objective(trial):
        params = {}
        for i, f in enumerate(features):
            params[f] = [
                0.0, 0.0,
                trial.suggest_float(f"{f}_x2", 0.0, max_x_values[i]), trial.suggest_float(f"{f}_y2", 0.0, 1.0),
                trial.suggest_float(f"{f}_x3", 0.0, max_x_values[i]), trial.suggest_float(f"{f}_y3", 0.0, 1.0),
                trial.suggest_float(f"{f}_x4", 0.0, max_x_values[i]), 0.0,
            ]
        mapped_list = [add_temporal_features(params, features, g, c, prediction_horizon) for (g, c) in train_data]
        X_all = pd.concat([df[df.columns.difference(features_to_remove)] for df in mapped_list], ignore_index=True)
        y_all = pd.concat([df[f"glucose_{prediction_horizon}"] for df in mapped_list], ignore_index=True)
        full_df = X_all.copy()
        target_col = f"glucose_{prediction_horizon}"
        full_df[target_col] = y_all.values
        idx_train, idx_val = train_test_split(range(len(full_df)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        rmse = train_and_predict(full_df, idx_train, idx_val, target_col, X_all.columns, full_df.iloc[idx_val])
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS, show_progress_bar=True)
    best = {f: [0.0, 0.0,
                study.best_params[f"{f}_x2"], study.best_params[f"{f}_y2"],
                study.best_params[f"{f}_x3"], study.best_params[f"{f}_y3"],
                study.best_params[f"{f}_x4"], 0.0] for f in features}
    json.dump(best, open(f"results/bezier_params/{approach_name}_bezier_params.json", 'w'), indent=2)
    return best
