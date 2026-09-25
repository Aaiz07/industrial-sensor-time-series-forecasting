# ============================================================
# STEP 20: WALK-FORWARD VALIDATION + FINAL MODEL SELECTION
# ============================================================
#
# Purpose:
#   Evaluate forecasting strategies across multiple chronological
#   windows instead of relying on one train/validation/test split.
#
# Models:
#   1. Persistence
#   2. Ridge Regression
#   3. XGBoost
#
# Target horizon:
#   Exactly 60 seconds into the future
#
# Sensors:
#   vib_x_rms
#   vib_y_rms
#   vib_z_rms
#   mag_x
#   mag_y
#   mag_z
#   temperature
#
# Output:
#   outputs/reports/step20_walk_forward_results.csv
#   outputs/reports/step20_model_summary.csv
#   outputs/reports/step20_final_model_selection.csv
#   outputs/forecasts/step20_walk_forward_predictions.csv
#
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\Vision\Desktop\Time_series"

CLEANED_DATA = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "sensor_data_cleaned.csv"
)

FUTURE_DATA = os.path.join(
    BASE_DIR,
    "outputs",
    "data",
    "future_features_60s.csv"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "reports"
)

FORECAST_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "forecasts"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGETS = [
    "vib_x_rms",
    "vib_y_rms",
    "vib_z_rms",
    "mag_x",
    "mag_y",
    "mag_z",
    "temperature"
]

FUTURE_TARGETS = {
    "vib_x_rms": "future_vib_x_rms",
    "vib_y_rms": "future_vib_y_rms",
    "vib_z_rms": "future_vib_z_rms",
    "mag_x": "future_mag_x",
    "mag_y": "future_mag_y",
    "mag_z": "future_mag_z",
    "temperature": "future_temperature"
}

HORIZON_SECONDS = 60

# Number of walk-forward folds.
# Each fold moves forward in time.
N_FOLDS = 5

# Initial training fraction.
# Example:
# 0.50 means first 50% of chronological data is used
# for the first training window.
INITIAL_TRAIN_FRACTION = 0.50

# Validation window fraction.
# The remaining data is divided into chronological folds.
VALIDATION_FRACTION = 0.10


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate standard regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def safe_r2(y_true, y_pred):
    """
    R2 can fail if a validation window has almost no variance.
    """

    try:
        return r2_score(y_true, y_pred)
    except Exception:
        return np.nan


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 20: WALK-FORWARD VALIDATION + FINAL MODEL SELECTION")
print("=" * 70)

print("\n[1/9] Loading data...")

if not os.path.exists(CLEANED_DATA):
    raise FileNotFoundError(
        f"Cleaned data not found:\n{CLEANED_DATA}"
    )

if not os.path.exists(FUTURE_DATA):
    raise FileNotFoundError(
        f"Future feature data not found:\n{FUTURE_DATA}"
    )

cleaned = pd.read_csv(CLEANED_DATA)
future = pd.read_csv(FUTURE_DATA)

cleaned["ts"] = pd.to_datetime(cleaned["ts"])
future["ts"] = pd.to_datetime(future["ts"])

cleaned = cleaned.sort_values("ts").reset_index(drop=True)
future = future.sort_values("ts").reset_index(drop=True)

print(f"Cleaned data shape : {cleaned.shape}")
print(f"Future data shape  : {future.shape}")


# ============================================================
# 5. VERIFY FUTURE TARGETS
# ============================================================

print("\n[2/9] Verifying future targets...")

missing_targets = [
    col for col in FUTURE_TARGETS.values()
    if col not in future.columns
]

if missing_targets:
    raise ValueError(
        f"Missing future target columns: {missing_targets}"
    )

print("All future target columns found.")

if "actual_horizon_seconds" in future.columns:

    horizon_values = (
        future["actual_horizon_seconds"]
        .dropna()
        .unique()
    )

    print(
        "Actual horizon values:",
        horizon_values[:20]
    )

    if not np.allclose(
        horizon_values,
        HORIZON_SECONDS
    ):
        raise ValueError(
            "Future target horizon is not exactly 60 seconds."
        )

    print("PASS: Exact 60-second horizon confirmed.")


# ============================================================
# 6. MERGE CURRENT SENSOR VALUES
# ============================================================

print("\n[3/9] Merging current sensor values...")

current_sensor_columns = [
    "ts"
] + TARGETS

current_data = cleaned[current_sensor_columns].copy()

df = future.merge(
    current_data,
    on="ts",
    how="inner",
    suffixes=("", "_current")
)

print(f"Rows after merge: {len(df):,}")

if len(df) == 0:
    raise ValueError(
        "Merge produced zero rows."
    )


# ============================================================
# 7. IDENTIFY MODEL FEATURES
# ============================================================

print("\n[4/9] Preparing model features...")

# Remove columns that must never be model inputs.

EXCLUDE_COLUMNS = [
    "ts",
    "target_ts",
    "actual_horizon_seconds"
]

# Future target columns must NEVER be features.
EXCLUDE_COLUMNS += list(
    FUTURE_TARGETS.values()
)

# Current raw sensor values must also be excluded.
# Historical versions of these values should only enter
# through lag/rolling features already contained in future_data.

EXCLUDE_COLUMNS += TARGETS

EXCLUDE_COLUMNS = list(
    dict.fromkeys(EXCLUDE_COLUMNS)
)

feature_columns = [
    col for col in df.columns
    if col not in EXCLUDE_COLUMNS
]

# Keep only numeric feature columns.
numeric_features = []

for col in feature_columns:

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):
        numeric_features.append(col)

feature_columns = numeric_features

print(
    f"Number of model features: "
    f"{len(feature_columns)}"
)

print("\nFirst 20 features:")

for feature in feature_columns[:20]:
    print("  ", feature)


# ============================================================
# 8. LEAKAGE CHECK
# ============================================================

print("\n[5/9] Running leakage checks...")

leakage_columns = []

for feature in feature_columns:

    # Future targets
    for future_target in FUTURE_TARGETS.values():

        if feature == future_target:
            leakage_columns.append(feature)

    # Raw current sensor values
    for target in TARGETS:

        if feature == target:
            leakage_columns.append(feature)

# Explicit future-looking names
future_keywords = [
    "future_",
    "target_"
]

for feature in feature_columns:

    feature_lower = feature.lower()

    for keyword in future_keywords:

        if feature_lower.startswith(keyword):

            # Some historical feature names may contain
            # "target" legitimately, but future_* should
            # definitely not appear.
            if feature not in leakage_columns:
                leakage_columns.append(feature)

if leakage_columns:

    raise ValueError(
        "Potential leakage detected:\n"
        + "\n".join(leakage_columns)
    )

print("PASS: No obvious future-target leakage detected.")


# ============================================================
# 9. CLEAN NUMERIC DATA
# ============================================================

print("\n[6/9] Cleaning model data...")

required_columns = (
    feature_columns
    + list(FUTURE_TARGETS.values())
    + TARGETS
)

df_model = df[
    ["ts"] + required_columns
].copy()

df_model = df_model.replace(
    [np.inf, -np.inf],
    np.nan
)

before_rows = len(df_model)

df_model = df_model.dropna(
    subset=required_columns
).reset_index(drop=True)

after_rows = len(df_model)

print(
    f"Rows removed due to missing values: "
    f"{before_rows - after_rows:,}"
)

print(
    f"Final usable rows: {after_rows:,}"
)


# ============================================================
# 10. CHRONOLOGICAL WALK-FORWARD FOLDS
# ============================================================

print("\n[7/9] Creating chronological walk-forward folds...")

n = len(df_model)

initial_train_size = int(
    n * INITIAL_TRAIN_FRACTION
)

remaining_size = n - initial_train_size

fold_size = int(
    remaining_size / N_FOLDS
)

if fold_size <= 0:
    raise ValueError(
        "Fold size is zero. "
        "Reduce N_FOLDS or provide more data."
    )

folds = []

for fold in range(N_FOLDS):

    train_end = (
        initial_train_size
        + fold * fold_size
    )

    validation_start = train_end

    if fold == N_FOLDS - 1:

        validation_end = n

    else:

        validation_end = (
            train_end + fold_size
        )

    if validation_end <= validation_start:
        continue

    folds.append(
        {
            "fold": fold + 1,
            "train_start": 0,
            "train_end": train_end,
            "val_start": validation_start,
            "val_end": validation_end
        }
    )

print(f"\nNumber of folds: {len(folds)}")

for fold_info in folds:

    fold_no = fold_info["fold"]

    train_start = fold_info["train_start"]
    train_end = fold_info["train_end"]

    val_start = fold_info["val_start"]
    val_end = fold_info["val_end"]

    print(
        f"\nFold {fold_no}"
    )

    print(
        f"  Train: "
        f"{df_model.iloc[train_start]['ts']} "
        f"-> "
        f"{df_model.iloc[train_end - 1]['ts']}"
    )

    print(
        f"  Validation: "
        f"{df_model.iloc[val_start]['ts']} "
        f"-> "
        f"{df_model.iloc[val_end - 1]['ts']}"
    )

    print(
        f"  Train rows: {train_end - train_start:,}"
    )

    print(
        f"  Validation rows: "
        f"{val_end - val_start:,}"
    )


# ============================================================
# 11. WALK-FORWARD EVALUATION
# ============================================================

print("\n[8/9] Running walk-forward evaluation...")

all_results = []
all_predictions = []

for target in TARGETS:

    future_target = FUTURE_TARGETS[target]

    print("\n" + "=" * 70)
    print(f"SENSOR: {target}")
    print("=" * 70)

    X = df_model[feature_columns].values

    y = df_model[
        future_target
    ].values

    current = df_model[
        target
    ].values

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Ridge": Ridge(
            alpha=10.0
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }

    # --------------------------------------------------------
    # Fold loop
    # --------------------------------------------------------

    for fold_info in folds:

        fold_no = fold_info["fold"]

        train_start = fold_info["train_start"]
        train_end = fold_info["train_end"]

        val_start = fold_info["val_start"]
        val_end = fold_info["val_end"]

        X_train = X[
            train_start:train_end
        ]

        y_train = y[
            train_start:train_end
        ]

        X_val = X[
            val_start:val_end
        ]

        y_val = y[
            val_start:val_end
        ]

        current_val = current[
            val_start:val_end
        ]

        timestamps_val = df_model[
            "ts"
        ].iloc[
            val_start:val_end
        ].values

        # ====================================================
        # Persistence baseline
        # ====================================================

        persistence_pred = current_val.copy()

        mae, rmse, r2 = calculate_metrics(
            y_val,
            persistence_pred
        )

        all_results.append(
            {
                "sensor": target,
                "fold": fold_no,
                "model": "Persistence",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "train_rows": len(X_train),
                "validation_rows": len(X_val)
            }
        )

        for i in range(len(y_val)):

            all_predictions.append(
                {
                    "sensor": target,
                    "fold": fold_no,
                    "model": "Persistence",
                    "ts": timestamps_val[i],
                    "actual": y_val[i],
                    "prediction": persistence_pred[i]
                }
            )

        # ====================================================
        # Learned models
        # ====================================================

        for model_name, model in models.items():

            print(
                f"  Fold {fold_no}: "
                f"{target} -> {model_name}"
            )

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_val
            )

            mae, rmse, r2 = calculate_metrics(
                y_val,
                prediction
            )

            all_results.append(
                {
                    "sensor": target,
                    "fold": fold_no,
                    "model": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "train_rows": len(X_train),
                    "validation_rows": len(X_val)
                }
            )

            for i in range(len(y_val)):

                all_predictions.append(
                    {
                        "sensor": target,
                        "fold": fold_no,
                        "model": model_name,
                        "ts": timestamps_val[i],
                        "actual": y_val[i],
                        "prediction": prediction[i]
                    }
                )


# ============================================================
# 12. SAVE FOLD RESULTS
# ============================================================

print("\n[9/9] Saving results...")

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)

results_path = os.path.join(
    REPORT_DIR,
    "step20_walk_forward_results.csv"
)

predictions_path = os.path.join(
    FORECAST_DIR,
    "step20_walk_forward_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print(
    f"\nSaved fold results:\n"
    f"{results_path}"
)

print(
    f"Saved predictions:\n"
    f"{predictions_path}"
)


# ============================================================
# 13. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WALK-FORWARD MODEL SUMMARY")
print("=" * 70)

summary = (
    results_df
    .groupby(
        ["sensor", "model"]
    )
    .agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        std_rmse=("rmse", "std"),
        mean_r2=("r2", "mean"),
        std_r2=("r2", "std"),
        folds=("fold", "count")
    )
    .reset_index()
)

summary_path = os.path.join(
    REPORT_DIR,
    "step20_model_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    "\nAverage performance across folds:"
)

for sensor in TARGETS:

    print("\n" + "-" * 70)
    print(sensor)

    sensor_summary = summary[
        summary["sensor"] == sensor
    ].sort_values(
        "mean_rmse"
    )

    print(
        sensor_summary[
            [
                "model",
                "mean_rmse",
                "std_rmse",
                "mean_r2",
                "std_r2"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 14. SELECT BEST MODEL BY WALK-FORWARD RMSE
# ============================================================

best_models = []

for sensor in TARGETS:

    sensor_summary = summary[
        summary["sensor"] == sensor
    ].copy()

    sensor_summary = sensor_summary.sort_values(
        by=[
            "mean_rmse",
            "std_rmse"
        ]
    )

    best = sensor_summary.iloc[0]

    best_models.append(
        {
            "sensor": sensor,
            "best_model": best["model"],
            "mean_mae": best["mean_mae"],
            "std_mae": best["std_mae"],
            "mean_rmse": best["mean_rmse"],
            "std_rmse": best["std_rmse"],
            "mean_r2": best["mean_r2"],
            "std_r2": best["std_r2"],
            "folds": best["folds"]
        }
    )

best_models_df = pd.DataFrame(
    best_models
)

final_selection_path = os.path.join(
    REPORT_DIR,
    "step20_final_model_selection.csv"
)

best_models_df.to_csv(
    final_selection_path,
    index=False
)


# ============================================================
# 15. COMPARE LEARNED MODELS AGAINST PERSISTENCE
# ============================================================

print("\n" + "=" * 70)
print("IMPROVEMENT OVER PERSISTENCE")
print("=" * 70)

improvement_rows = []

for sensor in TARGETS:

    sensor_summary = summary[
        summary["sensor"] == sensor
    ]

    persistence = sensor_summary[
        sensor_summary["model"] == "Persistence"
    ]

    if persistence.empty:
        continue

    persistence_rmse = (
        persistence["mean_rmse"].iloc[0]
    )

    persistence_r2 = (
        persistence["mean_r2"].iloc[0]
    )

    for _, row in sensor_summary.iterrows():

        rmse_improvement = (
            (
                persistence_rmse
                - row["mean_rmse"]
            )
            / persistence_rmse
        ) * 100

        r2_difference = (
            row["mean_r2"]
            - persistence_r2
        )

        improvement_rows.append(
            {
                "sensor": sensor,
                "model": row["model"],
                "persistence_mean_rmse":
                    persistence_rmse,
                "model_mean_rmse":
                    row["mean_rmse"],
                "rmse_improvement_percent":
                    rmse_improvement,
                "persistence_mean_r2":
                    persistence_r2,
                "model_mean_r2":
                    row["mean_r2"],
                "r2_difference":
                    r2_difference
            }
        )

improvement_df = pd.DataFrame(
    improvement_rows
)

improvement_path = os.path.join(
    REPORT_DIR,
    "step20_improvement_over_persistence.csv"
)

improvement_df.to_csv(
    improvement_path,
    index=False
)


# ============================================================
# 16. PRINT FINAL MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION FROM WALK-FORWARD VALIDATION")
print("=" * 70)

print(
    best_models_df.to_string(
        index=False
    )
)


# ============================================================
# 17. INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
The purpose of Step 20 is not to maximize one test score.

Instead, we ask:

1. Does a model perform consistently across time?
2. Does it beat the persistence baseline?
3. Is the improvement stable?
4. Does R2 remain meaningful across folds?
5. Does model performance collapse when the operating regime changes?

A model with slightly worse average RMSE but much more stable
performance may be preferable to a model with one excellent fold
and several poor folds.

Important:
A negative R2 does NOT automatically mean the model is useless
for every application. It means that, for that evaluation window,
the model performs worse than predicting the mean target value.

For forecasting, we should additionally compare against the
persistence baseline because persistence is a strong baseline for
slow-moving sensor signals.
"""
)


# ============================================================
# 18. FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 20 COMPLETE")
print("=" * 70)

print(
    "\nFiles created:"
)

print(
    f"1. {results_path}"
)

print(
    f"2. {summary_path}"
)

print(
    f"3. {final_selection_path}"
)

print(
    f"4. {improvement_path}"
)

print(
    f"5. {predictions_path}"
)

print(
    "\nNext:"
)

print(
    "Review the walk-forward results before training another model."
)

print(
    "The final model should be selected based on consistency "
    "across multiple chronological folds, not one test split."
)

print(
    "\nDone."
)