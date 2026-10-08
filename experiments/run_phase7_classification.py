import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.battery.loading import load_battery_data, create_summary_dataframe
from src.battery.targets import add_next_cycle_capacity_target
from src.battery.splits import leave_one_battery_out
from src.battery.features import get_feature_columns

def main():
    data_dir = project_root / "data" / "raw"
    battery_ids = ["B0005", "B0006", "B0007", "B0018"]
    
    all_summaries = []
    for bid in battery_ids:
        file_path = data_dir / f"{bid}.mat"
        if not file_path.exists():
            continue
        ops = load_battery_data(str(file_path), bid)
        df_summary = create_summary_dataframe(ops)
        
        df_discharge = df_summary[(df_summary["operation_type"] == "discharge") & (df_summary["capacity"].notna())].copy()
        all_summaries.append(df_discharge)
        
    df_all = pd.concat(all_summaries, ignore_index=True)
    
    df_with_targets = add_next_cycle_capacity_target(df_all)
    
    features = get_feature_columns()
    
    eol_threshold = 1.4
    
    # Binary target
    df_with_targets["next_cycle_below_eol"] = (df_with_targets["target_next_capacity"] <= eol_threshold).astype(int)
    
    cost_fn = 5.0
    cost_fp = 1.0
    
    metric_results = []
    prediction_results = []
    calibration_results = []
    
    for held_out_battery in battery_ids:
        train_df, test_df = leave_one_battery_out(df_with_targets, held_out_battery)
        
        X_train = train_df[features]
        y_train_reg = train_df["target_next_capacity"]
        y_train_clf = train_df["next_cycle_below_eol"]
        
        X_test = test_df[features]
        y_test_reg = test_df["target_next_capacity"]
        y_test_clf = test_df["next_cycle_below_eol"]
        
        clf_pipeline = Pipeline([
            ("scaler", StandardScaler()), 
            ("svc", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))
        ])
        
        reg_pipeline = Pipeline([
            ("scaler", StandardScaler()), 
            ("ridge", Ridge(alpha=1.0))
        ])
        
        clf_pipeline.fit(X_train, y_train_clf)
        reg_pipeline.fit(X_train, y_train_reg)
        
        y_pred_prob = clf_pipeline.predict_proba(X_test)[:, 1]
        y_pred_reg = reg_pipeline.predict(X_test)
        
        y_pred_bin_50 = (y_pred_prob >= 0.5).astype(int)
        y_pred_bin_cost = (y_pred_prob >= (1.0 / 6.0)).astype(int)
        y_pred_bin_ridge = (y_pred_reg <= 1.4).astype(int)
        
        conditions = [
            ("direct_svc_threshold_0.5", y_pred_bin_50, "direct_classification", y_pred_prob, np.nan),
            ("direct_svc_cost_threshold", y_pred_bin_cost, "direct_classification", y_pred_prob, np.nan),
            ("ridge_regress_then_threshold", y_pred_bin_ridge, "regress_then_threshold", np.nan, y_pred_reg)
        ]
        
        for cond_name, y_pred_bin, approach, prob, reg_val in conditions:
            tn, fp, fn, tp = confusion_matrix(y_test_clf, y_pred_bin, labels=[0, 1]).ravel()
            weighted_cost = fp * cost_fp + fn * cost_fn
            weighted_cost_per_row = weighted_cost / len(y_test_clf) if len(y_test_clf) > 0 else 0
            
            metric_results.append({
                "battery": held_out_battery,
                "approach": approach,
                "condition": cond_name,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "accuracy": accuracy_score(y_test_clf, y_pred_bin),
                "precision": precision_score(y_test_clf, y_pred_bin, zero_division=0),
                "recall": recall_score(y_test_clf, y_pred_bin, zero_division=0),
                "f1": f1_score(y_test_clf, y_pred_bin, zero_division=0),
                "weighted_cost": weighted_cost,
                "weighted_cost_per_row": weighted_cost_per_row
            })
            
            for i in range(len(test_df)):
                idx = test_df.index[i]
                
                # Check for np.nan logic
                p_prob = prob[i] if isinstance(prob, np.ndarray) else np.nan
                p_reg = reg_val[i] if isinstance(reg_val, np.ndarray) else np.nan
                
                prediction_results.append({
                    "battery_id": held_out_battery,
                    "discharge_cycle_index": test_df.loc[idx, "discharge_cycle_index"],
                    "actual_next_capacity": y_test_reg.iloc[i],
                    "actual_binary_label": y_test_clf.iloc[i],
                    "approach": approach,
                    "condition": cond_name,
                    "predicted_probability": p_prob,
                    "predicted_next_capacity": p_reg,
                    "predicted_binary_label": y_pred_bin[i]
                })
                
        # Calibration bins
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for bl, bu in bins:
            if bu == 1.0:
                mask = (y_pred_prob >= bl) & (y_pred_prob <= bu)
            else:
                mask = (y_pred_prob >= bl) & (y_pred_prob < bu)
                
            if mask.sum() > 0:
                n_samples = mask.sum()
                mean_prob = y_pred_prob[mask].mean()
                obs_rate = y_test_clf.iloc[mask].mean()
                
                calibration_results.append({
                    "held_out_battery": held_out_battery,
                    "bin_lower": bl,
                    "bin_upper": bu,
                    "n_samples": n_samples,
                    "mean_predicted_probability": mean_prob,
                    "observed_positive_rate": obs_rate
                })
                
    out_dir = project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_metrics = pd.DataFrame(metric_results)
    df_metrics.to_csv(out_dir / "classification_lobo_results.csv", index=False)
    
    df_preds = pd.DataFrame(prediction_results)
    df_preds.to_csv(out_dir / "classification_lobo_predictions.csv", index=False)
    
    if calibration_results:
        df_calib = pd.DataFrame(calibration_results)
        df_calib.to_csv(out_dir / "classification_lobo_calibration.csv", index=False)
        
    print("Phase 7 Experiment completed.")

if __name__ == "__main__":
    main()
