
import os
import shutil


def _resolve_lgbm_device():
    """Resolve LightGBM device from env and runtime availability.

    Env override:
      - LGBM_DEVICE=cpu
      - LGBM_DEVICE=gpu
      - LGBM_DEVICE=auto (default)
    """
    requested = os.getenv("LGBM_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "gpu"}:
        return requested
    # auto: use GPU only when NVIDIA runtime seems available
    return "gpu" if shutil.which("nvidia-smi") else "cpu"


LGBM_DEVICE = _resolve_lgbm_device()

# Patient IDs
PATIENTS_D1NAMO = ['001', '002', '004', '006', '007', '008']
PATIENTS_AZT1D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# Prediction horizons (in 5-minute intervals)
PREDICTION_HORIZONS = [6, 9, 12, 18, 24]  # 30min, 45min, 60min, 90min, 120min
PH_COLUMNS = [f'glucose_{h}' for h in PREDICTION_HORIZONS]

# Feature sets for optimization
OPTIMIZATION_FEATURES_D1NAMO = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
OPTIMIZATION_FEATURES_AZT1D = ['carbohydrates', 'insulin', 'correction']

# LightGBM parameters
LGB_PARAMS = {
    'random_state': 42,
    'deterministic': True,
    'num_threads': 8,
    'n_estimators': 100,
    'learning_rate': 0.1,
    'reg_lambda': 10,
    'verbosity': -1,
    'data_sample_strategy': 'goss',
    'max_depth': 3,
}

if LGBM_DEVICE == "gpu":
    # If GPU backend is unavailable in the installed LightGBM build,
    # processing_functions.train_and_predict will automatically fall back to CPU.
    LGB_PARAMS['device'] = 'gpu'

# Monotone constraints mapping for feature names
MONOTONE_MAP = {
    'simple_sugars': 1,
    'complex_sugars': 1,
    'carbohydrates': 1,
    'dietary_fibers': -1,
    'insulin': -1,
    'correction': -1,
}

# Common feature sets to remove during prediction
FEATURES_TO_REMOVE_D1NAMO = ['datetime', 'hour', 'patient_id', '_segment_id'] + PH_COLUMNS
FEATURES_TO_REMOVE_AZT1D = ['datetime', 'hour', 'patient', '_segment_id'] + PH_COLUMNS

# Default prediction horizon for analysis (60 minutes)
DEFAULT_PREDICTION_HORIZON = 12

# Glucose conversion factor (mmol/L to mg/dL)
GLUCOSE_CONVERSION_FACTOR = 18.0182

# File paths
D1NAMO_DATA_PATH = "diabetes_subset_pictures-glucose-food-insulin"
AZT1D_DATA_PATH = "AZT1D/CGM Records"
FOOD_DATA_PATH = "food_data/pixtral-large-latest"
RESULTS_PATH = "results"

# AZT1D column mappings
AZT1D_COLUMNS = {
    'datetime': 'EventDateTime',
    'glucose': 'CGM',
    'carbohydrates': 'CarbSize',
    'insulin': 'TotalBolusInsulinDelivered',
    'correction': 'CorrectionDelivered'
}

# Training and evaluation parameters
FAST_FEATURES = ['simple_sugars', 'insulin']
LOAD_PARAMS = True
N_TRIALS = 500
RANDOM_SEED = 42
N_JOBS = -1
CURRENT_PATIENT_WEIGHT = 10
VALIDATION_SIZE = 0.2
MAX_X_VALUES_FAST = 4.0
MAX_X_VALUES_SLOW = 8.0
STEP_SIZE = 12
SEGMENT_GAP_THRESHOLD_MINUTES = 6.0
HISTORY_WINDOW_POINTS = 6
