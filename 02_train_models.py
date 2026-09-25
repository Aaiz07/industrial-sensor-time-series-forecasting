"""
02_train_models.py

Improved, memory-safe time-series training pipeline.

Purpose:
- Train strong baselines + XGBoost using lag/rolling features.
- Train a compact multi-output LSTM.
- Evaluate every sensor independently.
- Save predictions, metrics, models, and plots.
- Keep the process leakage-free: scaler is fitted on TRAIN only.

Run:
    python 02_train_models.py
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    BASE_DIR, "outputs", "data", "sensor_data_cleaned.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR, PLOT_DIR]:
    os.makedirs(folder, exist_ok=True)

TIMESTAMP_COL = "ts"

SENSOR_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Compact lag set: captures immediate + medium-term behavior
LAGS = [1, 2, 3, 5, 10, 20, 30, 60]

# Rolling windows are deliberately modest.
ROLLING_WINDOWS = [5, 15, 30, 60]

LOOKBACK = 60

RANDOM_STATE = 42

# ============================================================
# REPRODUCIBILITY / CPU MEMORY
# ============================================================

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("=" * 70)
print("IMPROVED TIME-SERIES MODEL TRAINING")
print("=" * 70)

# ============================================================
# HELPERS
# ============================================================

def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-8
    if not np.any(mask):
        return 0.0
    return float(
        np.mean(
            2.0 * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        ) * 100.0
    )


def calculate_metrics(y_true, y_pred, sensor):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan

    return {
        "sensor": sensor,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "sMAPE_percent": float(smape(y_true, y_pred)),
    }


def inverse_one_column(pred_scaled, scaler, column_index):
    """
    Correct inverse transformation for one standardized sensor.
    """
    return (
        np.asarray(pred_scaled, dtype=np.float64) * scaler.scale_[column_index]
        + scaler.mean_[column_index]
    )


def make_lag_features(df, sensors):
    """
    Build leakage-free features from PAST values only.

    Features:
    - requested lags
    - rolling mean/std/min/max based on shifted values
    - first differences
    - time-of-day cyclic features
    """
    out = pd.DataFrame(index=df.index)

    # Past sensor values
    for sensor in sensors:
        for lag in LAGS:
            out[f"{sensor}_lag_{lag}"] = df[sensor].shift(lag)

        shifted = df[sensor].shift(1)

        for window in ROLLING_WINDOWS:
            out[f"{sensor}_roll_mean_{window}"] = (
                shifted.rolling(window).mean()
            )
            out[f"{sensor}_roll_std_{window}"] = (
                shifted.rolling(window).std()
            )
            out[f"{sensor}_roll_min_{window}"] = (
                shifted.rolling(window).min()
            )
            out[f"{sensor}_roll_max_{window}"] = (
                shifted.rolling(window).max()
            )

        out[f"{sensor}_diff_1"] = df[sensor].shift(1).diff()

    # Time features known in advance and therefore safe for forecasting
    ts = pd.to_datetime(df[TIMESTAMP_COL])

    seconds = (
        ts.dt.hour * 3600
        + ts.dt.minute * 60
        + ts.dt.second
    )

    day_seconds = 24 * 60 * 60

    out["hour_sin"] = np.sin(2 * np.pi * seconds / day_seconds)
    out["hour_cos"] = np.cos(2 * np.pi * seconds / day_seconds)

    out["minute_sin"] = np.sin(
        2 * np.pi * (ts.dt.minute * 60 + ts.dt.second) / 3600
    )
    out["minute_cos"] = np.cos(
        2 * np.pi * (ts.dt.minute * 60 + ts.dt.second) / 3600
    )

    return out


def plot_actual_predicted(y_true, y_pred, sensor, output_path, n_points=2000):
    n = min(n_points, len(y_true))

    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:n], label="Actual")
    plt.plot(y_pred[:n], label="Predicted")
    plt.title(f"{sensor} - Actual vs Predicted")
    plt.xlabel("Test sample")
    plt.ylabel(sensor)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/8] Loading cleaned data...")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{DATA_FILE}\n"
        "Run 01_data_pipeline.py first."
    )

df = pd.read_csv(DATA_FILE)

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce"
)

df = df.dropna(subset=[TIMESTAMP_COL]).copy()
df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

for col in SENSOR_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)

print(f"Rows: {len(df):,}")
print(f"Sensors: {len(SENSOR_COLS)}")

# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

print("\n[2/8] Creating chronological train/validation/test split...")

n = len(df)

train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(val_df):,}")
print(f"Test:       {len(test_df):,}")

# ============================================================
# SCALER - FIT TRAIN ONLY
# ============================================================

print("\n[3/8] Fitting scaler on TRAIN only...")

scaler = StandardScaler()
scaler.fit(train_df[SENSOR_COLS].astype(np.float32))

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "sensor_scaler.joblib")
)

# ============================================================
# IMPROVED XGBOOST
# ============================================================

print("\n[4/8] Building lag + rolling features for XGBoost...")

features_all = make_lag_features(df, SENSOR_COLS)

# Align target and features.
# Every feature uses shift(), so no current/future target information enters X.
xgb_data = pd.concat(
    [
        df[[TIMESTAMP_COL] + SENSOR_COLS],
        features_all
    ],
    axis=1
)

xgb_data = xgb_data.dropna().reset_index(drop=True)

feature_cols = [
    c for c in xgb_data.columns
    if c not in [TIMESTAMP_COL] + SENSOR_COLS
]

feature_count = len(feature_cols)

print(f"XGBoost feature count: {feature_count}")

# Determine split positions by timestamp rather than relying on
# the reduced dataframe retaining original row numbers.
train_last_ts = train_df[TIMESTAMP_COL].iloc[-1]
val_last_ts = val_df[TIMESTAMP_COL].iloc[-1]

xgb_train = xgb_data[xgb_data[TIMESTAMP_COL] <= train_last_ts].copy()
xgb_val = xgb_data[
    (xgb_data[TIMESTAMP_COL] > train_last_ts)
    & (xgb_data[TIMESTAMP_COL] <= val_last_ts)
].copy()
xgb_test = xgb_data[
    xgb_data[TIMESTAMP_COL] > val_last_ts
].copy()

X_train_xgb = xgb_train[feature_cols].astype(np.float32)
X_val_xgb = xgb_val[feature_cols].astype(np.float32)
X_test_xgb = xgb_test[feature_cols].astype(np.float32)

xgb_predictions = {}
xgb_metrics = []

for sensor in SENSOR_COLS:
    print(f"\nTraining XGBoost: {sensor}")

    model = XGBRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=2,
        random_state=RANDOM_STATE,
        eval_metric="rmse",
    )

    # Train directly on original units.
    # This avoids the previous inverse-scaling failure mode.
    y_train = xgb_train[sensor].astype(np.float32)
    y_val = xgb_val[sensor].astype(np.float32)

    model.fit(
        X_train_xgb,
        y_train,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False,
    )

    pred = model.predict(X_test_xgb).astype(np.float64)

    # Safety check: numerical predictions must be finite.
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"Non-finite XGBoost predictions for {sensor}")

    xgb_predictions[sensor] = pred

    y_test = xgb_test[sensor].to_numpy(dtype=np.float64)

    metrics = calculate_metrics(
        y_test,
        pred,
        sensor
    )
    metrics["model"] = "XGBoost_LagRolling"
    xgb_metrics.append(metrics)

    model_path = os.path.join(
        MODEL_DIR,
        f"xgb_{sensor}.joblib"
    )
    joblib.dump(model, model_path)

    print(
        f"  MAE={metrics['MAE']:.6f} | "
        f"RMSE={metrics['RMSE']:.6f} | "
        f"R2={metrics['R2']:.6f} | "
        f"sMAPE={metrics['sMAPE_percent']:.3f}%"
    )

# ============================================================
# PERSISTENCE BASELINE
# ============================================================

print("\n[5/8] Evaluating persistence baseline...")

# For every test timestamp, use the immediately previous observed value.
# This is a very important benchmark for time-series forecasting.
persistence_predictions = {}
persistence_metrics = []

for sensor in SENSOR_COLS:
    test_indices = xgb_test.index

    # xgb_data contains lag_1, but directly use the original lag feature.
    lag1_col = f"{sensor}_lag_1"

    pred = xgb_test[lag1_col].to_numpy(dtype=np.float64)
    y_test = xgb_test[sensor].to_numpy(dtype=np.float64)

    persistence_predictions[sensor] = pred

    metrics = calculate_metrics(
        y_test,
        pred,
        sensor
    )
    metrics["model"] = "Persistence"
    persistence_metrics.append(metrics)

# ============================================================
# LSTM DATA
# ============================================================

print("\n[6/8] Preparing memory-efficient LSTM sequences...")

# Scale using train-fitted scaler.
scaled_all = scaler.transform(
    df[SENSOR_COLS].astype(np.float32)
).astype(np.float32)

# Build windows over the complete chronological series.
# This is safe because each target only sees preceding observations.
X_seq = []
y_seq = []
target_indices = []

for i in range(LOOKBACK, len(scaled_all)):
    X_seq.append(scaled_all[i - LOOKBACK:i])
    y_seq.append(scaled_all[i])
    target_indices.append(i)

X_seq = np.asarray(X_seq, dtype=np.float32)
y_seq = np.asarray(y_seq, dtype=np.float32)
target_indices = np.asarray(target_indices)

train_mask = target_indices < train_end
val_mask = (
    (target_indices >= train_end)
    & (target_indices < val_end)
)
test_mask = target_indices >= val_end

X_train_lstm = X_seq[train_mask]
y_train_lstm = y_seq[train_mask]

X_val_lstm = X_seq[val_mask]
y_val_lstm = y_seq[val_mask]

X_test_lstm = X_seq[test_mask]
y_test_lstm = y_seq[test_mask]

print(f"LSTM train windows: {len(X_train_lstm):,}")
print(f"LSTM val windows:   {len(X_val_lstm):,}")
print(f"LSTM test windows:  {len(X_test_lstm):,}")

# ============================================================
# COMPACT MULTI-OUTPUT LSTM
# ============================================================

print("\nTraining compact multi-output LSTM...")

tf.keras.backend.clear_session()

lstm_model = Sequential([
    Input(shape=(LOOKBACK, len(SENSOR_COLS))),
    LSTM(48, return_sequences=True),
    Dropout(0.15),
    LSTM(24),
    Dropout(0.10),
    Dense(32, activation="relu"),
    Dense(len(SENSOR_COLS)),
])

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)

lstm_model.summary()

checkpoint_path = os.path.join(
    MODEL_DIR,
    "best_lstm.keras"
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    ),
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

history = lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

# Load best checkpoint explicitly
if os.path.exists(checkpoint_path):
    lstm_model = tf.keras.models.load_model(checkpoint_path)

lstm_pred_scaled = lstm_model.predict(
    X_test_lstm,
    batch_size=64,
    verbose=0
).astype(np.float64)

lstm_pred_original = scaler.inverse_transform(
    lstm_pred_scaled
)

y_test_lstm_original = scaler.inverse_transform(
    y_test_lstm.astype(np.float64)
)

lstm_metrics = []

for i, sensor in enumerate(SENSOR_COLS):
    metrics = calculate_metrics(
        y_test_lstm_original[:, i],
        lstm_pred_original[:, i],
        sensor
    )
    metrics["model"] = "LSTM"
    lstm_metrics.append(metrics)

    print(
        f"LSTM {sensor}: "
        f"MAE={metrics['MAE']:.6f} | "
        f"RMSE={metrics['RMSE']:.6f} | "
        f"R2={metrics['R2']:.6f} | "
        f"sMAPE={metrics['sMAPE_percent']:.3f}%"
    )

# Save training history
pd.DataFrame(history.history).to_csv(
    os.path.join(REPORT_DIR, "lstm_training_history.csv"),
    index=False
)

# ============================================================
# ALIGN PREDICTIONS
# ============================================================

print("\n[7/8] Selecting the best model per sensor...")

# Use LSTM test timestamps for the common evaluation area.
# XGB loses the first 60 rows due to lag/rolling features.
# Both test areas are chronological and overlap almost completely.
prediction_df = pd.DataFrame({
    TIMESTAMP_COL: df.iloc[val_end + LOOKBACK:][TIMESTAMP_COL].values
})

# Safer construction: use LSTM test targets as the primary common test set.
prediction_df = pd.DataFrame({
    TIMESTAMP_COL: df.iloc[target_indices[test_mask]][TIMESTAMP_COL].values
})

for i, sensor in enumerate(SENSOR_COLS):
    prediction_df[f"{sensor}_actual"] = y_test_lstm_original[:, i]
    prediction_df[f"{sensor}_lstm"] = lstm_pred_original[:, i]

# XGBoost and persistence predictions are aligned by timestamp.
xgb_lookup = xgb_test[[TIMESTAMP_COL]].copy()

for sensor in SENSOR_COLS:
    xgb_lookup[f"{sensor}_xgb"] = xgb_predictions[sensor]
    xgb_lookup[f"{sensor}_persistence"] = persistence_predictions[sensor]

prediction_df = prediction_df.merge(
    xgb_lookup,
    on=TIMESTAMP_COL,
    how="left"
)

# Collect all metrics
all_metrics = (
    persistence_metrics
    + xgb_metrics
    + lstm_metrics
)

metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    os.path.join(REPORT_DIR, "model_comparison.csv"),
    index=False
)

# Best model:
# Primary criterion = RMSE, with MAE as secondary consideration.
best_models = {}

for sensor in SENSOR_COLS:
    sensor_metrics = metrics_df[
        metrics_df["sensor"] == sensor
    ].copy()

    sensor_metrics = sensor_metrics.sort_values(
        ["RMSE", "MAE"],
        ascending=True
    )

    best_models[sensor] = sensor_metrics.iloc[0]["model"]

print("\nBEST MODEL PER SENSOR")
print("-" * 70)

for sensor, model_name in best_models.items():
    print(f"{sensor:15s} -> {model_name}")

# ============================================================
# BUILD BEST PREDICTION COLUMNS
# ============================================================

best_prediction_df = pd.DataFrame({
    TIMESTAMP_COL: prediction_df[TIMESTAMP_COL]
})

for sensor in SENSOR_COLS:
    best_prediction_df[f"{sensor}_actual"] = (
        prediction_df[f"{sensor}_actual"]
    )

    model_name = best_models[sensor]

    if model_name == "LSTM":
        best_prediction_df[f"{sensor}_predicted"] = (
            prediction_df[f"{sensor}_lstm"]
        )

    elif model_name == "XGBoost_LagRolling":
        best_prediction_df[f"{sensor}_predicted"] = (
            prediction_df[f"{sensor}_xgb"]
        )

    else:
        best_prediction_df[f"{sensor}_predicted"] = (
            prediction_df[f"{sensor}_persistence"]
        )

best_prediction_df.to_csv(
    os.path.join(DATA_DIR, "test_predictions_improved.csv"),
    index=False
)

# ============================================================
# PLOTS
# ============================================================

print("\nSaving actual-vs-predicted plots...")

for sensor in SENSOR_COLS:
    y_true = best_prediction_df[
        f"{sensor}_actual"
    ].to_numpy()

    y_pred = best_prediction_df[
        f"{sensor}_predicted"
    ].to_numpy()

    plot_actual_predicted(
        y_true,
        y_pred,
        sensor,
        os.path.join(
            PLOT_DIR,
            f"{sensor}_actual_vs_predicted_improved.png"
        )
    )

# ============================================================
# CONFIG / SUMMARY
# ============================================================

config = {
    "data_file": DATA_FILE,
    "rows": int(len(df)),
    "sensors": SENSOR_COLS,
    "train_ratio": TRAIN_RATIO,
    "validation_ratio": VAL_RATIO,
    "test_ratio": 1.0 - TRAIN_RATIO - VAL_RATIO,
    "lookback": LOOKBACK,
    "lags": LAGS,
    "rolling_windows": ROLLING_WINDOWS,
    "xgb_estimators": 350,
    "xgb_max_depth": 5,
    "xgb_learning_rate": 0.05,
    "lstm_units": [48, 24],
    "batch_size": 64,
    "random_state": RANDOM_STATE,
    "best_models": best_models,
}

with open(
    os.path.join(REPORT_DIR, "training_config.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(config, f, indent=4)

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")
print(
    "  outputs/data/test_predictions_improved.csv"
)
print(
    "  outputs/reports/model_comparison.csv"
)
print(
    "  outputs/reports/lstm_training_history.csv"
)
print(
    "  outputs/reports/training_config.json"
)
print(
    "  outputs/models/"
)
print(
    "  outputs/plots/*_actual_vs_predicted_improved.png"
)

print("\nIMPORTANT:")
print("Do NOT run 03_forecast_and_evaluate.py yet.")
print("First inspect model_comparison.csv and the new predictions.")
