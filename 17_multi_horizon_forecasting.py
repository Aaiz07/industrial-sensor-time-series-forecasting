# ============================================================
# 17_multi_horizon_forecasting.py
#
# Multi-Horizon True Forecasting Analysis
#
# Horizons:
#   1 minute
#   5 minutes
#   10 minutes
#   30 minutes
#   60 minutes
#
# Purpose:
#   Determine how predictable each sensor is at different
#   future horizons.
#
# IMPORTANT:
#   This is a forecasting diagnostic.
#   It does NOT train large/deep models.
#
#   Features at time t -> target at time t + horizon
#
#   No future values are used as input features.
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLEAN_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "plots"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


TIMESTAMP_COL = "ts"

SENSORS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"
]


# ------------------------------------------------------------
# Forecast horizons
# ------------------------------------------------------------

HORIZONS = {
    "1min": 60,
    "5min": 300,
    "10min": 600,
    "30min": 1800,
    "60min": 3600
}


# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------

# Use historical information only.
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
# HELPER FUNCTIONS
# ============================================================

def print_section(title):

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def calculate_rmse(y_true, y_pred):

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def calculate_mae(y_true, y_pred):

    return np.mean(
        np.abs(
            y_true - y_pred
        )
    )


def calculate_r2(y_true, y_pred):

    denominator = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if denominator == 0:
        return np.nan

    return (
        1
        -
        np.sum(
            (y_true - y_pred) ** 2
        )
        /
        denominator
    )


def safe_filename(name):

    return (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
    )


# ============================================================
# 1. CHECK INPUT
# ============================================================

print_section(
    "STEP 17: MULTI-HORIZON FORECASTING"
)

if not os.path.exists(CLEAN_FILE):

    raise FileNotFoundError(
        f"Input file not found:\n{CLEAN_FILE}"
    )

print(
    f"Input file:\n{CLEAN_FILE}"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    CLEAN_FILE
)

print(
    f"Rows loaded: {len(df):,}"
)

required_columns = [
    TIMESTAMP_COL
] + SENSORS

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns:\n"
        f"{missing_columns}"
    )


# ============================================================
# 3. CLEAN / SORT
# ============================================================

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL],
    errors="coerce"
)

for sensor in SENSORS:

    df[sensor] = pd.to_numeric(
        df[sensor],
        errors="coerce"
    )

df = df.dropna(
    subset=[TIMESTAMP_COL]
).copy()

df = df.sort_values(
    TIMESTAMP_COL
).reset_index(
    drop=True
)

print(
    f"Rows after timestamp validation: "
    f"{len(df):,}"
)

print(
    f"Start: {df[TIMESTAMP_COL].min()}"
)

print(
    f"End:   {df[TIMESTAMP_COL].max()}"
)


# ============================================================
# 4. SAMPLING INFORMATION
# ============================================================

print_section(
    "SAMPLING INFORMATION"
)

df["gap_seconds"] = (
    df[TIMESTAMP_COL]
    .diff()
    .dt.total_seconds()
)

median_gap = (
    df["gap_seconds"]
    .dropna()
    .median()
)

mean_gap = (
    df["gap_seconds"]
    .dropna()
    .mean()
)

print(
    f"Median sampling interval: "
    f"{median_gap:.4f} seconds"
)

print(
    f"Mean sampling interval: "
    f"{mean_gap:.4f} seconds"
)

print(
    f"Maximum sampling gap: "
    f"{df['gap_seconds'].max():.2f} seconds"
)


# ============================================================
# 5. CREATE HISTORICAL FEATURES
# ============================================================

print_section(
    "CREATING HISTORICAL FEATURES"
)

# All features here represent information available
# at or before the current timestamp.

for sensor in SENSORS:

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in LAGS:

        df[
            f"{sensor}_lag_{lag}"
        ] = (
            df[sensor]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    df[
        f"{sensor}_diff_1"
    ] = (
        df[sensor]
        -
        df[sensor].shift(1)
    )

    # --------------------------------------------------------
    # Rolling statistics
    #
    # Shift by one row BEFORE rolling so the current
    # observation is not accidentally included.
    # --------------------------------------------------------

    shifted = df[sensor].shift(1)

    for window in ROLLING_WINDOWS:

        df[
            f"{sensor}_rollmean_{window}"
        ] = (
            shifted
            .rolling(window)
            .mean()
        )

        df[
            f"{sensor}_rollstd_{window}"
        ] = (
            shifted
            .rolling(window)
            .std()
        )


# ============================================================
# 6. TIME FEATURES
# ============================================================

print_section(
    "CREATING TIME FEATURES"
)

df["hour_decimal"] = (
    df[TIMESTAMP_COL].dt.hour
    +
    df[TIMESTAMP_COL].dt.minute / 60.0
    +
    df[TIMESTAMP_COL].dt.second / 3600.0
)

df["time_sin"] = np.sin(
    2 * np.pi
    * df["hour_decimal"]
    / 24.0
)

df["time_cos"] = np.cos(
    2 * np.pi
    * df["hour_decimal"]
    / 24.0
)


# ============================================================
# 7. DEFINE FEATURES
# ============================================================

feature_columns = []

for sensor in SENSORS:

    for lag in LAGS:

        feature_columns.append(
            f"{sensor}_lag_{lag}"
        )

    feature_columns.append(
        f"{sensor}_diff_1"
    )

    for window in ROLLING_WINDOWS:

        feature_columns.append(
            f"{sensor}_rollmean_{window}"
        )

        feature_columns.append(
            f"{sensor}_rollstd_{window}"
        )

feature_columns += [
    "time_sin",
    "time_cos"
]

feature_columns = [
    col
    for col in feature_columns
    if col in df.columns
]

print(
    f"Total historical features: "
    f"{len(feature_columns)}"
)


# ============================================================
# 8. LEAKAGE CHECK
# ============================================================

print_section(
    "FEATURE LEAKAGE CHECK"
)

for sensor in SENSORS:

    if sensor in feature_columns:

        raise ValueError(
            f"Current raw sensor value found "
            f"directly as a feature: {sensor}"
        )

future_keywords = [
    "future_",
    "target_"
]

leakage_features = []

for feature in feature_columns:

    for keyword in future_keywords:

        if keyword in feature:

            leakage_features.append(
                feature
            )

if leakage_features:

    raise ValueError(
        "Potential future leakage detected:\n"
        + "\n".join(
            leakage_features
        )
    )

print(
    "Current raw target columns used directly: NO"
)

print(
    "Future target columns used: NO"
)

print(
    "LEAKAGE CHECK: PASS"
)


# ============================================================
# 9. CREATE EXACT FUTURE TARGETS
# ============================================================

print_section(
    "CREATING EXACT FUTURE TARGETS"
)

# We use timestamp lookup rather than row shifting.
#
# Example:
#
# Current time:
#   10:00:00
#
# 5-minute target:
#   10:05:00
#
# If that exact timestamp does not exist,
# the sample is not used.
#
# This prevents an irregular sampling interval from
# being incorrectly interpreted as an exact time horizon.

timestamp_lookup = (
    df[
        [TIMESTAMP_COL]
        + SENSORS
    ]
    .drop_duplicates(
        subset=[TIMESTAMP_COL]
    )
    .set_index(
        TIMESTAMP_COL
    )
)

results = []

prediction_records = []


# ============================================================
# 10. PROCESS EACH HORIZON
# ============================================================

for horizon_name, horizon_seconds in HORIZONS.items():

    print_section(
        f"HORIZON: {horizon_name} "
        f"({horizon_seconds} seconds)"
    )

    target_timestamp = (
        df[TIMESTAMP_COL]
        +
        pd.to_timedelta(
            horizon_seconds,
            unit="s"
        )
    )

    # --------------------------------------------------------
    # Exact timestamp matching
    # --------------------------------------------------------

    future_values = (
        timestamp_lookup
        .reindex(
            target_timestamp
        )
    )

    future_values.index = (
        df.index
    )

    # --------------------------------------------------------
    # Build horizon dataset
    # --------------------------------------------------------

    horizon_df = df[
        [
            TIMESTAMP_COL
        ]
        + feature_columns
        + SENSORS
    ].copy()

    horizon_df[
        "target_ts"
    ] = target_timestamp

    horizon_df[
        "actual_horizon_seconds"
    ] = horizon_seconds

    # Add future targets
    for sensor in SENSORS:

        horizon_df[
            f"future_{sensor}"
        ] = future_values[
            sensor
        ].values

    # --------------------------------------------------------
    # Drop unavailable future targets
    # --------------------------------------------------------

    future_columns = [
        f"future_{sensor}"
        for sensor in SENSORS
    ]

    horizon_df = horizon_df.dropna(
        subset=(
            feature_columns
            + future_columns
        )
    ).reset_index(
        drop=True
    )

    print(
        f"Exact future matches: "
        f"{len(horizon_df):,}"
    )

    coverage = (
        len(horizon_df)
        /
        len(df)
        *
        100
    )

    print(
        f"Coverage: {coverage:.2f}%"
    )

    if len(horizon_df) < 100:

        print(
            "WARNING: Too few matched rows "
            "for reliable modeling."
        )

        continue


    # ========================================================
    # 11. CHRONOLOGICAL SPLIT
    # ========================================================

    n = len(horizon_df)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    train_df = horizon_df.iloc[
        :train_end
    ].copy()

    val_df = horizon_df.iloc[
        train_end:val_end
    ].copy()

    test_df = horizon_df.iloc[
        val_end:
    ].copy()

    print(
        f"Train:      {len(train_df):,}"
    )

    print(
        f"Validation: {len(val_df):,}"
    )

    print(
        f"Test:       {len(test_df):,}"
    )


    # ========================================================
    # 12. PREPARE FEATURES
    # ========================================================

    X_train = train_df[
        feature_columns
    ].values

    X_val = val_df[
        feature_columns
    ].values

    X_test = test_df[
        feature_columns
    ].values


    # --------------------------------------------------------
    # Fit scaler ONLY on training data
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_val_scaled = (
        scaler.transform(
            X_val
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )


    # ========================================================
    # 13. FORECAST EACH SENSOR
    # ========================================================

    for sensor in SENSORS:

        future_target = (
            f"future_{sensor}"
        )

        y_train = train_df[
            future_target
        ].values

        y_val = val_df[
            future_target
        ].values

        y_test = test_df[
            future_target
        ].values

        # ----------------------------------------------------
        # Persistence
        #
        # Forecast future value = current value
        # ----------------------------------------------------

        persistence_val = val_df[
            sensor
        ].values

        persistence_test = test_df[
            sensor
        ].values

        persistence_val_mae = (
            calculate_mae(
                y_val,
                persistence_val
            )
        )

        persistence_val_rmse = (
            calculate_rmse(
                y_val,
                persistence_val
            )
        )

        persistence_val_r2 = (
            calculate_r2(
                y_val,
                persistence_val
            )
        )

        persistence_test_mae = (
            calculate_mae(
                y_test,
                persistence_test
            )
        )

        persistence_test_rmse = (
            calculate_rmse(
                y_test,
                persistence_test
            )
        )

        persistence_test_r2 = (
            calculate_r2(
                y_test,
                persistence_test
            )
        )

        results.append(
            {
                "horizon":
                    horizon_name,

                "horizon_seconds":
                    horizon_seconds,

                "sensor":
                    sensor,

                "model":
                    "Persistence",

                "validation_MAE":
                    persistence_val_mae,

                "validation_RMSE":
                    persistence_val_rmse,

                "validation_R2":
                    persistence_val_r2,

                "test_MAE":
                    persistence_test_mae,

                "test_RMSE":
                    persistence_test_rmse,

                "test_R2":
                    persistence_test_r2,

                "matched_rows":
                    len(horizon_df)
            }
        )


        # ----------------------------------------------------
        # Ridge
        # ----------------------------------------------------

        ridge = Ridge(
            alpha=1.0
        )

        ridge.fit(
            X_train_scaled,
            y_train
        )

        ridge_val_pred = ridge.predict(
            X_val_scaled
        )

        ridge_test_pred = ridge.predict(
            X_test_scaled
        )

        ridge_val_mae = (
            calculate_mae(
                y_val,
                ridge_val_pred
            )
        )

        ridge_val_rmse = (
            calculate_rmse(
                y_val,
                ridge_val_pred
            )
        )

        ridge_val_r2 = (
            calculate_r2(
                y_val,
                ridge_val_pred
            )
        )

        ridge_test_mae = (
            calculate_mae(
                y_test,
                ridge_test_pred
            )
        )

        ridge_test_rmse = (
            calculate_rmse(
                y_test,
                ridge_test_pred
            )
        )

        ridge_test_r2 = (
            calculate_r2(
                y_test,
                ridge_test_pred
            )
        )

        results.append(
            {
                "horizon":
                    horizon_name,

                "horizon_seconds":
                    horizon_seconds,

                "sensor":
                    sensor,

                "model":
                    "Ridge",

                "validation_MAE":
                    ridge_val_mae,

                "validation_RMSE":
                    ridge_val_rmse,

                "validation_R2":
                    ridge_val_r2,

                "test_MAE":
                    ridge_test_mae,

                "test_RMSE":
                    ridge_test_rmse,

                "test_R2":
                    ridge_test_r2,

                "matched_rows":
                    len(horizon_df)
            }
        )


        # ----------------------------------------------------
        # Store test predictions
        # ----------------------------------------------------

        prediction_block = pd.DataFrame(
            {
                TIMESTAMP_COL:
                    test_df[
                        TIMESTAMP_COL
                    ].values,

                "target_ts":
                    test_df[
                        "target_ts"
                    ].values,

                "horizon":
                    horizon_name,

                "horizon_seconds":
                    horizon_seconds,

                "sensor":
                    sensor,

                "actual":
                    y_test,

                "persistence":
                    persistence_test,

                "ridge":
                    ridge_test_pred
            }
        )

        prediction_records.append(
            prediction_block
        )


        # ----------------------------------------------------
        # Print current result
        # ----------------------------------------------------

        print(
            f"\n{sensor}"
        )

        print(
            f"  Persistence:"
            f"  Val RMSE="
            f"{persistence_val_rmse:.6f}"
            f" | Test RMSE="
            f"{persistence_test_rmse:.6f}"
        )

        print(
            f"  Ridge:"
            f"        Val RMSE="
            f"{ridge_val_rmse:.6f}"
            f" | Test RMSE="
            f"{ridge_test_rmse:.6f}"
        )


# ============================================================
# 14. RESULTS DATAFRAME
# ============================================================

print_section(
    "MULTI-HORIZON RESULTS"
)

results_df = pd.DataFrame(
    results
)

if len(results_df) == 0:

    raise RuntimeError(
        "No forecasting results were produced."
    )

results_df = results_df.sort_values(
    [
        "sensor",
        "horizon_seconds",
        "validation_RMSE"
    ]
)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step17_multi_horizon_results.csv"
    ),
    index=False
)


# ============================================================
# 15. SELECT BEST MODEL AT EACH HORIZON
# ============================================================

print_section(
    "BEST MODEL BY SENSOR AND HORIZON"
)

best_rows = []

for sensor in SENSORS:

    for horizon_name, horizon_seconds in HORIZONS.items():

        subset = results_df[
            (
                results_df["sensor"]
                == sensor
            )
            &
            (
                results_df["horizon"]
                == horizon_name
            )
        ].copy()

        if len(subset) == 0:
            continue

        best_idx = (
            subset[
                "validation_RMSE"
            ].idxmin()
        )

        best = subset.loc[
            best_idx
        ]

        best_rows.append(
            {
                "sensor":
                    sensor,

                "horizon":
                    horizon_name,

                "horizon_seconds":
                    horizon_seconds,

                "best_model":
                    best["model"],

                "validation_RMSE":
                    best[
                        "validation_RMSE"
                    ],

                "validation_MAE":
                    best[
                        "validation_MAE"
                    ],

                "validation_R2":
                    best[
                        "validation_R2"
                    ],

                "test_RMSE":
                    best[
                        "test_RMSE"
                    ],

                "test_MAE":
                    best[
                        "test_MAE"
                    ],

                "test_R2":
                    best[
                        "test_R2"
                    ]
            }
        )

best_df = pd.DataFrame(
    best_rows
)

print(
    best_df.to_string(
        index=False
    )
)

best_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step17_best_models_by_horizon.csv"
    ),
    index=False
)


# ============================================================
# 16. CALCULATE FORECAST DEGRADATION
# ============================================================

print_section(
    "FORECAST PERFORMANCE CHANGE WITH HORIZON"
)

degradation_rows = []

for sensor in SENSORS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ].copy()

    for model in [
        "Persistence",
        "Ridge"
    ]:

        model_results = sensor_results[
            sensor_results["model"]
            == model
        ].sort_values(
            "horizon_seconds"
        )

        if len(model_results) == 0:
            continue

        first_rmse = (
            model_results.iloc[0][
                "test_RMSE"
            ]
        )

        for _, row in model_results.iterrows():

            current_rmse = row[
                "test_RMSE"
            ]

            increase_percent = (
                (
                    current_rmse
                    -
                    first_rmse
                )
                /
                (
                    abs(first_rmse)
                    + 1e-12
                )
                * 100
            )

            degradation_rows.append(
                {
                    "sensor":
                        sensor,

                    "model":
                        model,

                    "horizon":
                        row["horizon"],

                    "horizon_seconds":
                        row[
                            "horizon_seconds"
                        ],

                    "test_RMSE":
                        current_rmse,

                    "increase_vs_shortest_horizon_percent":
                        increase_percent
                }
            )

degradation_df = pd.DataFrame(
    degradation_rows
)

degradation_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step17_forecast_degradation.csv"
    ),
    index=False
)

print(
    degradation_df.to_string(
        index=False
    )
)


# ============================================================
# 17. CREATE RMSE PLOTS
# ============================================================

print_section(
    "CREATING PERFORMANCE PLOTS"
)

for sensor in SENSORS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ].copy()

    plt.figure(
        figsize=(10, 6)
    )

    for model in [
        "Persistence",
        "Ridge"
    ]:

        model_results = sensor_results[
            sensor_results["model"]
            == model
        ].sort_values(
            "horizon_seconds"
        )

        if len(model_results) == 0:
            continue

        plt.plot(
            model_results[
                "horizon_seconds"
            ],
            model_results[
                "test_RMSE"
            ],
            marker="o",
            label=model
        )

    plt.xlabel(
        "Forecast Horizon (seconds)"
    )

    plt.ylabel(
        "Test RMSE"
    )

    plt.title(
        f"{sensor}: Forecast Error vs Horizon"
    )

    plt.xticks(
        list(HORIZONS.values()),
        list(HORIZONS.keys())
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            (
                "step17_horizon_rmse_"
                + safe_filename(sensor)
                + ".png"
            )
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 18. R2 PLOTS
# ============================================================

for sensor in SENSORS:

    sensor_results = results_df[
        results_df["sensor"] == sensor
    ].copy()

    plt.figure(
        figsize=(10, 6)
    )

    for model in [
        "Persistence",
        "Ridge"
    ]:

        model_results = sensor_results[
            sensor_results["model"]
            == model
        ].sort_values(
            "horizon_seconds"
        )

        if len(model_results) == 0:
            continue

        plt.plot(
            model_results[
                "horizon_seconds"
            ],
            model_results[
                "test_R2"
            ],
            marker="o",
            label=model
        )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Forecast Horizon (seconds)"
    )

    plt.ylabel(
        "Test R²"
    )

    plt.title(
        f"{sensor}: R² vs Forecast Horizon"
    )

    plt.xticks(
        list(HORIZONS.values()),
        list(HORIZONS.keys())
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            (
                "step17_horizon_r2_"
                + safe_filename(sensor)
                + ".png"
            )
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

print_section(
    "SAVING TEST PREDICTIONS"
)

if prediction_records:

    predictions_df = pd.concat(
        prediction_records,
        ignore_index=True
    )

    predictions_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "step17_multi_horizon_test_predictions.csv"
        ),
        index=False
    )

    print(
        f"Prediction rows saved: "
        f"{len(predictions_df):,}"
    )


# ============================================================
# 20. CREATE SUMMARY TABLE
# ============================================================

print_section(
    "FORECASTING LIMIT SUMMARY"
)

summary_rows = []

for sensor in SENSORS:

    sensor_best = best_df[
        best_df["sensor"] == sensor
    ].copy()

    if len(sensor_best) == 0:
        continue

    # --------------------------------------------------------
    # Find the longest horizon where best validation R²
    # remains >= 0.
    # --------------------------------------------------------

    useful = sensor_best[
        sensor_best[
            "validation_R2"
        ] >= 0
    ]

    if len(useful) > 0:

        longest_useful_horizon = (
            useful[
                "horizon_seconds"
            ].max()
        )

        longest_useful_name = (
            useful.loc[
                useful[
                    "horizon_seconds"
                ].idxmax(),
                "horizon"
            ]
        )

    else:

        longest_useful_horizon = 0

        longest_useful_name = (
            "None"
        )

    # --------------------------------------------------------
    # Best overall test R2
    # --------------------------------------------------------

    best_test_idx = (
        sensor_best[
            "test_R2"
        ].idxmax()
    )

    best_test_row = (
        sensor_best.loc[
            best_test_idx
        ]
    )

    summary_rows.append(
        {
            "sensor":
                sensor,

            "longest_horizon_with_nonnegative_validation_R2_seconds":
                longest_useful_horizon,

            "longest_horizon_with_nonnegative_validation_R2":
                longest_useful_name,

            "best_test_R2":
                best_test_row[
                    "test_R2"
                ],

            "best_test_RMSE":
                best_test_row[
                    "test_RMSE"
                ],

            "best_model":
                best_test_row[
                    "best_model"
                ],

            "best_horizon":
                best_test_row[
                    "horizon"
                ]
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

print(
    summary_df.to_string(
        index=False
    )
)

summary_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "step17_forecasting_limit_summary.csv"
    ),
    index=False
)


# ============================================================
# 21. SAVE CONFIGURATION
# ============================================================

config = {

    "forecast_horizons_seconds":
        HORIZONS,

    "lags":
        LAGS,

    "rolling_windows":
        ROLLING_WINDOWS,

    "feature_count":
        len(feature_columns),

    "selection_rule":
        "Lowest validation RMSE",

    "test_used_for_model_selection":
        False,

    "target_alignment":
        "Exact timestamp matching",

    "leakage_check":
        "PASS",

    "input_file":
        CLEAN_FILE
}

with open(
    os.path.join(
        REPORT_DIR,
        "step17_config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4,
        default=str
    )


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print_section(
    "STEP 17 COMPLETED"
)

print(
    "Multi-horizon forecasting analysis completed."
)

print(
    "\nForecast horizons tested:"
)

for name, seconds in HORIZONS.items():

    print(
        f"  {name}: {seconds} seconds"
    )

print(
    "\nTarget alignment:"
)

print(
    "  Exact timestamp matching"
)

print(
    "  No row-shift approximation"
)

print(
    "  No future target leakage"
)

print(
    "\nReports saved:"
)

print(
    "  outputs/reports/"
    "step17_multi_horizon_results.csv"
)

print(
    "  outputs/reports/"
    "step17_best_models_by_horizon.csv"
)

print(
    "  outputs/reports/"
    "step17_forecast_degradation.csv"
)

print(
    "  outputs/reports/"
    "step17_multi_horizon_test_predictions.csv"
)

print(
    "  outputs/reports/"
    "step17_forecasting_limit_summary.csv"
)

print(
    "  outputs/reports/"
    "step17_config.json"
)

print(
    "\nPlots saved:"
)

print(
    "  outputs/plots/"
    "step17_horizon_rmse_*.png"
)

print(
    "  outputs/plots/"
    "step17_horizon_r2_*.png"
)

print(
    "\nIMPORTANT:"
)

print(
    "Do not train another model yet."
)

print(
    "First inspect the multi-horizon results."
)