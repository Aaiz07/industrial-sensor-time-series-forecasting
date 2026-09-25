# Industrial Sensor Time-Series Forecasting System

## 1. Project Overview

This project develops an end-to-end industrial sensor time-series forecasting system.

The system performs data validation, cleaning, exploratory analysis, feature engineering, future-target construction, model training, chronological validation, multi-horizon forecasting, real-time prediction, database storage, forecast monitoring, visualization, and system validation.

## 2. Problem Statement

The objective of the project is to determine whether historical industrial sensor measurements contain enough temporal information to predict future sensor values.

A major goal is not only to train forecasting models, but also to determine which sensor signals are genuinely forecastable under realistic future conditions.

## 3. Sensor Channels

The project contains seven sensor channels:

- vib_x_rms
- vib_y_rms
- vib_z_rms
- mag_x
- mag_y
- mag_z
- temperature

## 4. Dataset

The original dataset contains 59,230 observations.

The original columns are:

- ts
- vib_x_rms
- vib_y_rms
- vib_z_rms
- mag_x
- mag_y
- mag_z
- temperature

After duplicate timestamp handling, the cleaned dataset contains 59,195 observations.

## 5. Data Quality Analysis

The data-quality pipeline evaluates:

- Timestamp validity
- Duplicate timestamps
- Missing values
- Sampling intervals
- Temporal gaps
- Sensor distributions
- Extreme sensor measurements

Duplicate timestamps were aggregated using sensor-wise means rather than arbitrarily deleting measurements.

## 6. Sampling and Time Regularization

The raw data has irregular sampling intervals. The median sampling interval is approximately one second.

A one-second regularized timeline was investigated. The regularized timeline contains 86,399 timestamps.

The analysis identified a major outage in the data. Large outages were preserved rather than blindly interpolated.

## 7. Temporal Resolution Analysis

Different temporal resolutions were investigated because sensor predictability can depend on the aggregation level.

The analysis showed that vibration, magnetic, and temperature signals have very different temporal characteristics.

Aggregation can reduce noise and RMSE, but lower RMSE alone does not prove that a sensor is genuinely predictable.

## 8. Feature Engineering

The final forecasting pipeline uses 146 historical features.

Important feature groups include:

- Historical lag features
- First-difference features
- Cross-axis vibration features
- Magnetic lag features
- Rolling means
- Rolling standard deviations
- Time-of-day sine and cosine features

Rolling features use shifted historical values so that future observations cannot leak into the feature set.

## 9. Target Alignment

An important discovery during the project was the difference between current-value prediction and true future forecasting.

Early experiments produced extremely high R² scores because the target represented the current timestamp rather than a future timestamp.

The issue was diagnosed and corrected before final model evaluation.

## 10. True Future Targets

True future targets were created using exact timestamp alignment.

For the primary 60-second forecasting problem, each feature timestamp is matched with the sensor observation exactly 60 seconds in the future.

The resulting future-target dataset contains 41,881 matched observations.

## 11. Leakage Prevention

The final forecasting workflow includes explicit leakage checks.

The checks verify that:

- Current raw target values are not used as predictor features
- Lag-1 represents historical information
- Rolling features use shifted data
- Future target columns are excluded from model inputs
- Target timestamps occur after feature timestamps

## 12. Models Evaluated

The project investigated several forecasting approaches:

- Persistence baseline
- Moving-average baselines
- Ridge Regression
- XGBoost
- LSTM

More complex models were not automatically assumed to be better. Baselines were retained throughout the evaluation.

## 13. Chronological Validation

Random train-test splitting was avoided because it can produce misleading results for time-series forecasting.

The project uses chronological splitting and later performs five-fold walk-forward validation for final model selection.

## 14. Final Model Selection

Walk-forward validation selected the following models:

- vib_x_rms -> XGBoost
- vib_y_rms -> XGBoost
- vib_z_rms -> XGBoost
- mag_x -> Ridge
- mag_y -> Ridge
- mag_z -> Ridge
- temperature -> Persistence

A selected model should not automatically be interpreted as a highly accurate model. Selection means that it performed best relative to the alternatives under the validation procedure.

## 15. Final Walk-Forward Performance

The following results represent the primary generalization performance of the final forecasting system.

| Sensor | Final Model | MAE | RMSE | RMSE Std | R² |
|---|---|---:|---:|---:|---:|
| vib_x_rms | XGBoost | 0.019250 | 0.036500 | 0.031473 | -8.375215 |
| vib_y_rms | XGBoost | 0.024974 | 0.046564 | 0.038200 | -7.574728 |
| vib_z_rms | XGBoost | 0.008276 | 0.015306 | 0.011545 | -2.613671 |
| mag_x | Ridge | 114.777000 | 132.499000 | 0.995781 | -0.002038 |
| mag_y | Ridge | 114.555000 | 132.387000 | 1.089499 | -0.003333 |
| mag_z | Ridge | 113.912000 | 131.702000 | 0.559924 | -0.001523 |
| temperature | Persistence | 0.124780 | 0.189486 | 0.083086 | 0.817088 |

## 16. Sensor Group Performance

| Sensor Group | MAE | RMSE | R² |
|---|---:|---:|---:|
| Magnetic | 114.414667 | 132.196000 | -0.002298 |
| Temperature | 0.124780 | 0.189486 | 0.817088 |
| Vibration | 0.017500 | 0.032790 | -6.187871 |

## 17. Model Summary

| Model | Sensor / Group | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Persistence | Summary | 0.124780 | 0.189486 | 0.817088 |
| Ridge | Summary | 114.414667 | 132.196000 | -0.002298 |
| XGBoost | Summary | 0.017500 | 0.032790 | -6.187871 |

## 18. Distribution Shift

The project identified significant temporal distribution shift between training, validation, and test periods.

The vibration test period is substantially quieter and more stable than earlier portions of the dataset.

Temperature also operates in a different range during the test period.

This distribution shift is a major reason why vibration models that perform well during training can generalize poorly to future operating regimes.

## 19. Vibration Forecasting

XGBoost was selected for all three vibration channels through walk-forward validation.

However, the final walk-forward R² values remain negative.

Therefore, the vibration models should be interpreted as weak forecasting models under unseen operating regimes, rather than as reliable high-accuracy predictors.

## 20. Magnetic Forecasting

Ridge Regression was selected for the three magnetic channels.

The magnetic R² values remain approximately zero, indicating that the available historical features provide very limited point-forecasting information.

Magnetic predictions are therefore treated as low-confidence outputs in the monitoring system.

## 21. Temperature Forecasting

Temperature is the strongest forecastable sensor in the project.

Persistence provides strong performance because temperature changes gradually over short forecasting horizons.

The final walk-forward temperature R² is approximately 0.817, demonstrating useful future predictive performance.

## 22. Multi-Horizon Forecasting

The project evaluates exact future timestamps at:

- 60 seconds
- 300 seconds
- 600 seconds
- 1800 seconds
- 3600 seconds

Temperature remains useful through the tested 10-minute horizon, while its performance becomes poor at the tested 30-minute and 60-minute horizons.

| Sensor | Horizon | Model | R² |
|---|---:|---|---:|
| vib_x_rms | 60 sec | N/A | -4.257000 |
| vib_x_rms | 300 sec | N/A | -16.272000 |
| vib_x_rms | 600 sec | N/A | -101.171000 |
| vib_x_rms | 1800 sec | N/A | -132.697000 |
| vib_x_rms | 3600 sec | N/A | -64.384000 |
| vib_y_rms | 60 sec | N/A | -3.328000 |
| vib_y_rms | 300 sec | N/A | -10.406000 |
| vib_y_rms | 600 sec | N/A | -66.076000 |
| vib_y_rms | 1800 sec | N/A | -78.957000 |
| vib_y_rms | 3600 sec | N/A | -33.191000 |
| vib_z_rms | 60 sec | N/A | -1.132000 |
| vib_z_rms | 300 sec | N/A | -5.763000 |
| vib_z_rms | 600 sec | N/A | -25.026000 |
| vib_z_rms | 1800 sec | N/A | -33.614000 |
| vib_z_rms | 3600 sec | N/A | -20.429000 |
| mag_x | 60 sec | N/A | -0.004500 |
| mag_x | 300 sec | N/A | -0.002820 |
| mag_x | 600 sec | N/A | -0.001480 |
| mag_x | 1800 sec | N/A | -0.002410 |
| mag_x | 3600 sec | N/A | -0.002920 |
| mag_y | 60 sec | N/A | -0.003860 |
| mag_y | 300 sec | N/A | -0.000620 |
| mag_y | 600 sec | N/A | -0.001270 |
| mag_y | 1800 sec | N/A | -0.003590 |
| mag_y | 3600 sec | N/A | -0.011480 |
| mag_z | 60 sec | N/A | -0.003580 |
| mag_z | 300 sec | N/A | -0.003860 |
| mag_z | 600 sec | N/A | -0.004130 |
| mag_z | 1800 sec | N/A | -0.004040 |
| mag_z | 3600 sec | N/A | -0.006380 |
| temperature | 60 sec | N/A | 0.934887 |
| temperature | 300 sec | N/A | 0.716699 |
| temperature | 600 sec | N/A | 0.501596 |
| temperature | 1800 sec | N/A | -0.120017 |
| temperature | 3600 sec | N/A | -0.336376 |

## 23. Forecastability Conclusion

The project demonstrates that forecastability is sensor-specific.

**Temperature:** Strong short-term forecastability.

**Vibration:** Weak generalization under regime shift.

**Magnetic:** Very limited point forecastability with the available data.

This is an important result because a professional forecasting project should identify when a signal cannot be reliably predicted rather than forcing every sensor to show positive model performance.

## 24. Final Model Training

After walk-forward model selection, the selected model for each sensor was trained using the available labeled forecasting data and saved as a deployment artifact.

Training-set metrics are not used as unbiased forecasting performance estimates. Walk-forward results remain the main generalization metrics.

## 25. Real-Time Forecasting

The real-time forecasting pipeline reconstructs the exact 146-feature schema required by the final model artifacts.

The service generates forecasts 60 seconds ahead for all seven sensors.

## 26. Simulated Real-Time Forecasting

Historical observations are replayed sequentially to simulate a real-time sensor stream.

This verifies that the deployment pipeline can repeatedly construct features and generate future predictions.

## 27. Prediction Database

Forecasts are stored in a SQLite database located at:

`outputs/database/forecasting.db`

The `forecast_predictions` table stores one record per sensor forecast.

Important fields include:

- feature_timestamp
- forecast_timestamp
- horizon_seconds
- sensor
- model
- current_value
- predicted_value
- change_from_current
- created_at

## 28. Prediction Monitoring

Forecasts are passed through a monitoring layer.

Vibration monitoring uses warning and critical thresholds.

Temperature monitoring uses temperature thresholds.

Magnetic predictions are explicitly marked as low-confidence because their forecasting R² is approximately zero. Large magnetic percentage changes are therefore not automatically treated as reliable anomaly alerts.

## 29. Streamlit Dashboard

A Streamlit dashboard provides an interactive interface for viewing forecast predictions and monitoring results.

The dashboard reads directly from the SQLite forecasting database.

## 30. End-to-End Validation

The complete system was validated after integrating the forecasting service, database, and monitoring pipeline.

The end-to-end validation completed successfully and verified consistency between forecasting and monitoring records.

## 31. Final System Health Check

The final system health check verifies model artifacts, reports, database tables, predictions, monitoring outputs, and project components.

The temperature Persistence model intentionally does not require a trained machine-learning model object because the latest observation itself is used as the forecast.

## 32. System Architecture

```text
Raw Sensor Data
       |
       v
Data Validation
       |
       v
Cleaning & EDA
       |
       v
Sampling / Gap Analysis
       |
       v
Feature Engineering
       |
       v
Future Target Construction
       |
       v
Leakage Validation
       |
       v
Model Training
       |
       v
Walk-Forward Validation
       |
       v
Final Model Selection
       |
       v
Real-Time Forecasting
       |
       v
SQLite Prediction Database
       |
       v
Prediction Monitoring
       |
       v
Streamlit Dashboard
```

## 33. Professional Project Workflow

```text
REAL-WORLD PROBLEM
       |
       v
DEFINE TARGET
       |
       v
DATA
       |
       v
CLEANING + EDA
       |
       v
PREPROCESSING
       |
       v
TRAIN / VALIDATION / TEST
       |
       v
BASELINE
       |
       v
MODEL SELECTION
       |
       v
TRAINING
       |
       v
VALIDATION
       |
       v
ERROR ANALYSIS
       |
       v
IMPROVEMENT
       |
       v
FINAL MODEL
       |
       v
DEPLOYMENT
```

## 34. Key Lessons

The main lessons from this project are:

1. High training accuracy does not guarantee future forecasting accuracy.

2. Target alignment must be explicitly validated in time-series problems.

3. Random splitting can produce misleading forecasting results.

4. Persistence is an important forecasting baseline.

5. More complex models do not automatically solve distribution shift.

6. Different sensors can have fundamentally different forecastability.

7. Negative R² is a meaningful result when a model fails to beat a simple baseline.

8. Deployment monitoring should account for model reliability.

## 35. Project Limitations

- The dataset covers approximately one day.
- Operating regimes vary significantly through the day.
- Vibration generalization is weak under regime shift.
- Magnetic channels have very low point forecastability.
- External machine operating variables are unavailable.
- Current forecasts are point predictions rather than probabilistic forecasts.

## 36. Future Improvements

Future versions of the system could include:

- Longer historical datasets
- Multiple days or months of operation
- Machine RPM
- Machine load
- Operating-state information
- Maintenance events
- Regime-specific forecasting models
- Concept-drift detection
- Online retraining
- Probabilistic forecasting
- Prediction intervals
- Additional industrial context features

## 37. Important Project Components

```text
01_data_pipeline.py
02_train_models.py
03_forecast_and_evaluate.py
03_lag_diagnostics.py
04_time_regularization.py
05_validate_regularization.py
06_temporal_resolution.py
07_feature_predictability.py
08_horizon_analysis.py
09_final_feature_strategy.py
10_final_model_training.py
11_validate_target_alignment.py
12_create_future_targets.py
13_train_true_forecasting_models.py
14_error_distribution_analysis.py
15_regime_analysis.py
16_regime_aware_forecasting.py
17_multi_horizon_forecasting.py
18_robust_regime_evaluation.py
19_improve_features_forecasting.py
20_walk_forward_final_selection.py
21_final_model_training.py
22_final_analysis_and_visualization.py
23_realtime_forecasting.py
24_simulated_realtime.py
25_prediction_database.py
26_prediction_monitoring.py
27_forecasting_dashboard.py
28_realtime_forecasting_service.py
29_realtime_monitoring_service.py
30_end_to_end_validation.py
31_final_system_health_check.py
32_final_performance_report.py
33_final_visualizations.py
34_final_project_documentation.py
```

## 38. Output Structure

```text
outputs/
|
+-- data/
|
+-- forecasts/
|
+-- models/
|
+-- plots/
|
+-- reports/
|
+-- database/
    |
    +-- forecasting.db
```

## 39. Final Project Status

**Data pipeline:** Complete

**Feature engineering:** Complete

**Target alignment validation:** Complete

**True forecasting models:** Complete

**Walk-forward validation:** Complete

**Final model training:** Complete

**Real-time forecasting:** Complete

**SQLite database:** Complete

**Prediction monitoring:** Complete

**Streamlit dashboard:** Complete

**End-to-end validation:** Passed

**System health validation:** Passed

**Performance reporting:** Complete

**Final visualizations:** Complete

**Final documentation:** Complete

## 40. Final Conclusion

This project demonstrates a complete industrial sensor time-series forecasting workflow from raw data to an operational forecasting and monitoring system.

The strongest forecasting result is temperature, which shows useful short-term predictability and remains useful through the tested 10-minute horizon.

Magnetic channels demonstrate very limited point forecastability, while vibration forecasting is strongly affected by temporal regime shift.

These results demonstrate an important machine-learning principle: the objective of a forecasting project is not to force every model to produce a high accuracy score. The goal is to measure genuine future predictive ability, understand model limitations, and build a system that uses predictions according to their reliability.

The final project therefore combines forecasting research, time-series validation, deployment engineering, database integration, monitoring, and visualization into one end-to-end system.