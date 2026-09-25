"""
03_forecast_and_evaluate.py

FINAL FUTURE FORECASTING PIPELINE

Purpose:
- Load cleaned sensor data
- Build leakage-free historical features
- Forecast future sensor values
- Use exact timestamp-based future targets for evaluation
- Train final models
- Generate actual vs predicted plots
- Generate future forecasts from the latest available observation
- Save forecasts, metrics, models and plots

Forecast horizons:
    1 minute
    5 minutes
    10 minutes

Models:
    Persistence
    Ridge Regression

The model is selected using validation RMSE.
The test set is used only for final evaluation.

Run:
    python 03_forecast_and_evaluate.py
"""

import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
FORECAST_DIR = os.path.join(OUTPUT_DIR, "forecasts")

for folder in [
    DATA_DIR,
    MODEL_DIR,
    REPORT_DIR,
    PLOT_DIR,
    FORECAST_DIR
]:
    os.makedirs(folder, exist_ok=True)


TIMESTAMP_COL = "ts"

SENSOR_COLS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"
]

# Final practical forecasting horizons
HORIZONS = {
    "1min": 60,
    "5min": 300,
    "10min": 600
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_STATE = 42


# Historical features
LAGS = [
    1,
    2,
    5,
    10,
    30,
    60
]

ROLLING_WINDOWS = [
    5,
    15,
    30,
    60
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 75)
    print("LOADING CLEANED SENSOR DATA")
    print("=" * 75)

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"\nCould not find:\n{DATA_FILE}\n\n"
            "Run 01_data_pipeline.py first."
        )

    df = pd.read_csv(DATA_FILE)

    df[TIMESTAMP_COL] = pd.to_datetime(
        df[TIMESTAMP_COL],
        errors="coerce"
    )

    df = df.dropna(
        subset=[TIMESTAMP_COL]
    )

    df = df.sort_values(
        TIMESTAMP_COL
    )

    df = df.drop_duplicates(
        subset=[TIMESTAMP_COL],
        keep="first"
    )

    for sensor in SENSOR_COLS:

        df[sensor] = pd.to_numeric(
            df[sensor],
            errors="coerce"
        )

    df = df.dropna(
        subset=SENSOR_COLS
    )

    df = df.reset_index(drop=True)

    print(f"Rows       : {len(df):,}")
    print(
        f"Start time : {df[TIMESTAMP_COL].min()}"
    )
    print(
        f"End time   : {df[TIMESTAMP_COL].max()}"
    )

    return df


# ============================================================
# BUILD HISTORICAL FEATURES
# ============================================================

def build_features(df):

    print("\nBuilding historical features...")

    data = df.copy()

    feature_cols = []

    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    for sensor in SENSOR_COLS:

        for lag in LAGS:

            col = f"{sensor}_lag_{lag}"

            # IMPORTANT:
            # shift uses ONLY past observations.
            data[col] = data[sensor].shift(lag)

            feature_cols.append(col)

    # --------------------------------------------------------
    # DIFFERENCE FEATURES
    # --------------------------------------------------------

    for sensor in SENSOR_COLS:

        col = f"{sensor}_diff_1"

        data[col] = (
            data[sensor]
            - data[sensor].shift(1)
        )

        feature_cols.append(col)

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------

    for sensor in SENSOR_COLS:

        for window in ROLLING_WINDOWS:

            mean_col = (
                f"{sensor}_rolling_mean_{window}"
            )

            std_col = (
                f"{sensor}_rolling_std_{window}"
            )

            # shift(1) is critical.
            #
            # It prevents the current target
            # from entering the feature.

            shifted = data[sensor].shift(1)

            data[mean_col] = (
                shifted
                .rolling(window)
                .mean()
            )

            data[std_col] = (
                shifted
                .rolling(window)
                .std()
            )

            feature_cols.append(mean_col)
            feature_cols.append(std_col)

    # --------------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------------

    seconds = (
        data[TIMESTAMP_COL].dt.hour * 3600
        + data[TIMESTAMP_COL].dt.minute * 60
        + data[TIMESTAMP_COL].dt.second
    )

    data["time_sin"] = np.sin(
        2 * np.pi * seconds / 86400
    )

    data["time_cos"] = np.cos(
        2 * np.pi * seconds / 86400
    )

    feature_cols.extend([
        "time_sin",
        "time_cos"
    ])

    print(
        f"Features created: {len(feature_cols)}"
    )

    return data, feature_cols


# ============================================================
# CREATE EXACT FUTURE TARGET
# ============================================================

def create_future_dataset(
    feature_df,
    original_df,
    sensor,
    horizon_seconds
):

    current = feature_df[
        [TIMESTAMP_COL] +
        [
            c for c in feature_df.columns
            if c != TIMESTAMP_COL
        ]
    ].copy()

    future = original_df[
        [
            TIMESTAMP_COL,
            sensor
        ]
    ].copy()

    future = future.rename(
        columns={
            TIMESTAMP_COL: "future_ts",
            sensor: "target"
        }
    )

    current["future_ts"] = (
        current[TIMESTAMP_COL]
        + pd.Timedelta(
            seconds=horizon_seconds
        )
    )

    merged = current.merge(
        future,
        on="future_ts",
        how="inner"
    )

    merged = merged.drop(
        columns=["future_ts"]
    )

    return merged


# ============================================================
# PERSISTENCE MODEL
# ============================================================

def persistence_prediction(
    data,
    sensor
):

    return data[
        f"{sensor}_lag_1"
    ].to_numpy(dtype=float)


# ============================================================
# TRAIN RIDGE
# ============================================================

def train_ridge(
    X_train,
    y_train
):

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=10.0
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# TRAIN / VALIDATE / TEST
# ============================================================

def evaluate_sensor(
    data,
    sensor,
    feature_cols
):

    X = data[
        feature_cols
    ]

    y = data[
        "target"
    ]

    n = len(data)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[
        train_end:val_end
    ]

    y_val = y.iloc[
        train_end:val_end
    ]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    persistence_val = persistence_prediction(
        data.iloc[train_end:val_end],
        sensor
    )

    persistence_test = persistence_prediction(
        data.iloc[val_end:],
        sensor
    )

    val_mask = (
        np.isfinite(
            persistence_val
        )
        &
        np.isfinite(
            y_val.to_numpy()
        )
    )

    test_mask = (
        np.isfinite(
            persistence_test
        )
        &
        np.isfinite(
            y_test.to_numpy()
        )
    )

    persistence_val_metrics = calculate_metrics(
        y_val.to_numpy()[val_mask],
        persistence_val[val_mask]
    )

    persistence_test_metrics = calculate_metrics(
        y_test.to_numpy()[test_mask],
        persistence_test[test_mask]
    )

    # --------------------------------------------------------
    # RIDGE
    # --------------------------------------------------------

    ridge = train_ridge(
        X_train,
        y_train
    )

    ridge_val = ridge.predict(
        X_val
    )

    ridge_test = ridge.predict(
        X_test
    )

    ridge_val_metrics = calculate_metrics(
        y_val,
        ridge_val
    )

    ridge_test_metrics = calculate_metrics(
        y_test,
        ridge_test
    )

    # --------------------------------------------------------
    # SELECT MODEL USING VALIDATION
    # --------------------------------------------------------

    if (
        ridge_val_metrics["RMSE"]
        <
        persistence_val_metrics["RMSE"]
    ):

        best_model_name = "Ridge"

        best_val_metrics = ridge_val_metrics

        best_test_metrics = ridge_test_metrics

        best_test_prediction = ridge_test

    else:

        best_model_name = "Persistence"

        best_val_metrics = persistence_val_metrics

        best_test_metrics = persistence_test_metrics

        best_test_prediction = persistence_test

    return {
        "best_model_name": best_model_name,

        "best_val_metrics": best_val_metrics,

        "best_test_metrics": best_test_metrics,

        "ridge": ridge,

        "ridge_val": ridge_val,

        "ridge_test": ridge_test,

        "persistence_val": persistence_val,

        "persistence_test": persistence_test,

        "best_test_prediction": best_test_prediction,

        "test_timestamps": data.iloc[
            val_end:
        ][TIMESTAMP_COL].to_numpy(),

        "y_test": y_test.to_numpy()
    }


# ============================================================
# SAVE EVALUATION PLOT
# ============================================================

def save_evaluation_plot(
    timestamps,
    actual,
    predicted,
    sensor,
    horizon_name,
    metrics
):

    plt.figure(
        figsize=(15, 5)
    )

    plt.plot(
        timestamps,
        actual,
        label="Actual"
    )

    plt.plot(
        timestamps,
        predicted,
        label="Predicted"
    )

    plt.title(
        f"{sensor} - {horizon_name} Forecast"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        sensor
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    filename = os.path.join(
        PLOT_DIR,
        f"03_{sensor}_{horizon_name}_evaluation.png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()


# ============================================================
# FUTURE FORECAST
# ============================================================

def generate_future_forecast(
    df,
    feature_df,
    feature_cols,
    sensor,
    horizon_seconds,
    horizon_name,
    selected_model_name,
    ridge_model
):

    latest_row = feature_df.iloc[
        [-1]
    ].copy()

    latest_timestamp = latest_row[
        TIMESTAMP_COL
    ].iloc[0]

    # --------------------------------------------------------
    # Ridge forecast
    # --------------------------------------------------------

    if selected_model_name == "Ridge":

        X_latest = latest_row[
            feature_cols
        ]

        prediction = ridge_model.predict(
            X_latest
        )[0]

    # --------------------------------------------------------
    # Persistence forecast
    # --------------------------------------------------------

    else:

        prediction = latest_row[
            f"{sensor}_lag_1"
        ].iloc[0]

    future_timestamp = (
        latest_timestamp
        + pd.Timedelta(
            seconds=horizon_seconds
        )
    )

    return {
        "sensor": sensor,
        "forecast_horizon": horizon_name,
        "horizon_seconds": horizon_seconds,
        "last_observed_timestamp": latest_timestamp,
        "forecast_timestamp": future_timestamp,
        "model": selected_model_name,
        "forecast_value": float(prediction)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("FINAL INDUSTRIAL SENSOR FORECASTING")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_df, feature_cols = build_features(
        df
    )

    # --------------------------------------------------------
    # LEAKAGE CHECK
    # --------------------------------------------------------

    print("\nRunning leakage checks...")

    for sensor in SENSOR_COLS:

        if sensor in feature_cols:

            raise RuntimeError(
                f"LEAKAGE DETECTED: "
                f"current target {sensor} "
                f"is directly used as feature."
            )

    print("LEAKAGE CHECK: PASS")

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    all_results = []
    future_forecasts = []

    # --------------------------------------------------------
    # EACH HORIZON
    # --------------------------------------------------------

    for horizon_name, horizon_seconds in HORIZONS.items():

        print("\n")
        print("=" * 75)
        print(
            f"HORIZON: {horizon_name} "
            f"({horizon_seconds} seconds)"
        )
        print("=" * 75)

        for sensor in SENSOR_COLS:

            print("\n" + "-" * 65)
            print(
                f"SENSOR: {sensor}"
            )
            print("-" * 65)

            # ------------------------------------------------
            # CREATE EXACT FUTURE DATASET
            # ------------------------------------------------

            sensor_data = create_future_dataset(
                feature_df,
                df,
                sensor,
                horizon_seconds
            )

            required = feature_cols + [
                "target"
            ]

            sensor_data = sensor_data.dropna(
                subset=required
            ).reset_index(drop=True)

            if len(sensor_data) < 100:

                print(
                    "Not enough matched rows."
                )

                continue

            print(
                f"Matched rows: "
                f"{len(sensor_data):,}"
            )

            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            result = evaluate_sensor(
                sensor_data,
                sensor,
                feature_cols
            )

            best_model = result[
                "best_model_name"
            ]

            val_metrics = result[
                "best_val_metrics"
            ]

            test_metrics = result[
                "best_test_metrics"
            ]

            print(
                f"Best model: {best_model}"
            )

            print(
                f"Validation RMSE: "
                f"{val_metrics['RMSE']:.6f}"
            )

            print(
                f"Validation R2: "
                f"{val_metrics['R2']:.6f}"
            )

            print(
                f"Test MAE: "
                f"{test_metrics['MAE']:.6f}"
            )

            print(
                f"Test RMSE: "
                f"{test_metrics['RMSE']:.6f}"
            )

            print(
                f"Test R2: "
                f"{test_metrics['R2']:.6f}"
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            all_results.append({
                "sensor": sensor,
                "horizon": horizon_name,
                "horizon_seconds": horizon_seconds,
                "best_model": best_model,

                "validation_MAE":
                    val_metrics["MAE"],

                "validation_RMSE":
                    val_metrics["RMSE"],

                "validation_R2":
                    val_metrics["R2"],

                "test_MAE":
                    test_metrics["MAE"],

                "test_RMSE":
                    test_metrics["RMSE"],

                "test_R2":
                    test_metrics["R2"],

                "matched_rows":
                    len(sensor_data)
            })

            # ------------------------------------------------
            # PLOT
            # ------------------------------------------------

            save_evaluation_plot(
                result["test_timestamps"],
                result["y_test"],
                result["best_test_prediction"],
                sensor,
                horizon_name,
                test_metrics
            )

            # ------------------------------------------------
            # SAVE RIDGE MODEL
            # ------------------------------------------------

            if best_model == "Ridge":

                model_path = os.path.join(
                    MODEL_DIR,
                    f"03_{sensor}_{horizon_name}_ridge.joblib"
                )

                joblib.dump(
                    result["ridge"],
                    model_path
                )

            # ------------------------------------------------
            # FUTURE FORECAST
            # ------------------------------------------------

            forecast = generate_future_forecast(
                df,
                feature_df,
                feature_cols,
                sensor,
                horizon_seconds,
                horizon_name,
                best_model,
                result["ridge"]
            )

            future_forecasts.append(
                forecast
            )

            print(
                f"Future forecast: "
                f"{forecast['forecast_value']:.6f}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_file = os.path.join(
        REPORT_DIR,
        "03_final_forecasting_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    # --------------------------------------------------------
    # SAVE FUTURE FORECASTS
    # --------------------------------------------------------

    forecast_df = pd.DataFrame(
        future_forecasts
    )

    forecast_file = os.path.join(
        FORECAST_DIR,
        "03_future_sensor_forecasts.csv"
    )

    forecast_df.to_csv(
        forecast_file,
        index=False
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    summary = {
        "script":
            "03_forecast_and_evaluate.py",

        "dataset":
            DATA_FILE,

        "forecast_horizons":
            HORIZONS,

        "sensors":
            SENSOR_COLS,

        "feature_count":
            len(feature_cols),

        "train_ratio":
            TRAIN_RATIO,

        "validation_ratio":
            VAL_RATIO,

        "leakage_check":
            "PASS"
    }

    with open(
        os.path.join(
            REPORT_DIR,
            "03_forecasting_config.json"
        ),
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            default=str
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 75)
    print("FINAL FORECASTS")
    print("=" * 75)

    if len(forecast_df) > 0:

        display_cols = [
            "sensor",
            "forecast_horizon",
            "forecast_timestamp",
            "model",
            "forecast_value"
        ]

        print(
            forecast_df[
                display_cols
            ].to_string(
                index=False
            )
        )

    print("\n")
    print("=" * 75)
    print("FILES SAVED")
    print("=" * 75)

    print(
        f"Results:\n{results_file}"
    )

    print(
        f"\nFuture forecasts:\n{forecast_file}"
    )

    print(
        f"\nPlots:\n{PLOT_DIR}"
    )

    print(
        f"\nModels:\n{MODEL_DIR}"
    )

    print("\n")
    print("=" * 75)
    print("FORECASTING COMPLETED")
    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()