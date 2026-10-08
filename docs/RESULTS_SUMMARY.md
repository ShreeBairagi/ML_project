# Final Results Summary

This document provides a purely factual and descriptive summary of the numerical results generated during the experiments.

## Baselines
Performance of the baselines under Leave-One-Battery-Out (LOBO) evaluation:
- **Exponential Baseline (LOBO):**
  - B0005: MAE = 0.0248, RMSE = 0.0306, R² = 0.9737
  - B0006: MAE = 0.0272, RMSE = 0.0365, R² = 0.9786
  - B0007: MAE = 0.0194, RMSE = 0.0244, R² = 0.9766
  - B0018: MAE = 0.0240, RMSE = 0.0336, R² = 0.9516
- **Last-Observed-Value Baseline (LOBO):**
  - B0005: MAE = 0.0081, RMSE = 0.0133, R² = 0.9951
  - B0006: MAE = 0.0144, RMSE = 0.0236, R² = 0.9910
  - B0007: MAE = 0.0069, RMSE = 0.0124, R² = 0.9940
  - B0018: MAE = 0.0142, RMSE = 0.0226, R² = 0.9781

## Leakage Experiment
**Split Leakage:**
- Random Cycle Split (Wrong): MAE averaged around 0.0093, RMSE around 0.0167 across 5 seeds.
- LOBO (Correct): 
  - B0005: MAE = 0.0066, RMSE = 0.0128
  - B0006: MAE = 0.0130, RMSE = 0.0229
  - B0007: MAE = 0.0102, RMSE = 0.0153
  - B0018: MAE = 0.0145, RMSE = 0.0223

**Preprocessing Leakage (Random Split, seed 42):**
- Training-only Scaler (Correct): MAE = 0.011826, RMSE = 0.021681
- Global Scaler (Wrong): MAE = 0.011830, RMSE = 0.021681

## Single Models (LOBO)
Model performance ranges across the four held-out batteries:
- **LinearRegression / Ridge / Lasso / ElasticNet**: MAE ranges from ~0.0066 (B0005) to ~0.0145 (B0018).
- **DecisionTreeRegressor**: MAE ranges from ~0.0145 (B0018) to ~0.0185 (B0007).
- **KNeighborsRegressor**: MAE ranges from ~0.0097 (B0005) to ~0.0415 (B0007).
- **SVR**: MAE ranges from ~0.0082 (B0007) to ~0.0944 (B0006).
(Full results saved in `results/tables/single_model_lobo_results.csv`)

## Ensembles (LOBO)
Ensemble models showed the following ranges across held-out batteries:
- **BaggingRegressor**: MAE range ~0.0081 (B0005) to ~0.0163 (B0007).
- **RandomForestRegressor**: MAE range ~0.0069 (B0005) to ~0.0148 (B0007).
- **VotingRegressor**: MAE range ~0.0064 (B0005) to ~0.0254 (B0018).
- **SimpleAveraging**: MAE range ~0.0064 (B0005) to ~0.0254 (B0018).
- **WeightedAveraging**: MAE range ~0.0053 (B0005) to ~0.0248 (B0018).
- Residual correlation matrices for each battery are saved under `results/tables/residual_corr_*.csv`.

## Classification and Costs
Class counts: B0005 (119/44 TN/TP), B0006 (107/58), B0007 (145/0), B0018 (101/22).
*Note: B0007 has zero positive examples under the chosen EOL definition.*

- **Direct SVC (threshold 0.5):**
  - B0005: 44 TP, 119 TN, 4 FP, 0 FN (Cost: 4.0)
  - B0006: 58 TP, 107 TN, 1 FP, 1 FN (Cost: 6.0)
  - B0007: 0 TP, 145 TN, 22 FP, 0 FN (Cost: 22.0)
  - B0018: 22 TP, 101 TN, 2 FP, 6 FN (Cost: 32.0)
- **Direct SVC (cost-optimized threshold):**
  - B0005: Cost 9.0 (44 TP, 114 TN, 9 FP, 0 FN)
  - B0006: Cost 3.0 (59 TP, 105 TN, 3 FP, 0 FN)
  - B0007: Cost 29.0 (0 TP, 138 TN, 29 FP, 0 FN)
  - B0018: Cost 12.0 (26 TP, 101 TN, 2 FP, 2 FN)
- **Ridge (regress-then-threshold):**
  - B0005: Cost 0.0 (44 TP, 123 TN, 0 FP, 0 FN)
  - B0006: Cost 11.0 (57 TP, 107 TN, 1 FP, 2 FN)
  - B0007: Cost 7.0 (0 TP, 160 TN, 7 FP, 0 FN)
  - B0018: Cost 17.0 (25 TP, 101 TN, 2 FP, 3 FN)

Calibration outputs are saved at `results/tables/calibration_bins.csv`.

## Error Analysis
Error metrics by life region (early, middle, late) are logged in `results/tables/error_by_life_region.csv`.
The largest errors across all models are saved at `results/tables/largest_errors.csv`.

## Synthetic Shift
- **In-distribution**: MAE = 0.0116, RMSE = 0.0145, R² = 0.9951
- **Shifted**: MAE = 0.0223, RMSE = 0.0262, R² = 0.9931
- **Difference (Shifted - In-distribution)**: MAE = +0.0106, RMSE = +0.0117, R² = -0.0020
