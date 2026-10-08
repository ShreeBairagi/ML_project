import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.battery.loading import load_battery_data, create_summary_dataframe
from src.battery.targets import add_next_cycle_capacity_target
from src.battery.splits import leave_one_battery_out
from src.battery.features import get_feature_columns
from src.battery.ensembles import get_phase6_base_models, get_phase6_sklearn_ensembles, get_phase6_weights

def main():
    data_dir = project_root / "data" / "raw"
    battery_ids = ["B0005", "B0006", "B0007", "B0018"]
    
    # Load data
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
    target_col = "target_next_capacity"
    
    base_models = get_phase6_base_models()
    sklearn_ensembles = get_phase6_sklearn_ensembles()
    weights = get_phase6_weights()
    
    results = []
    predictions_log = []
    
    out_dir_tables = project_root / "results" / "tables"
    out_dir_tables.mkdir(parents=True, exist_ok=True)
    out_dir_figures = project_root / "results" / "figures"
    out_dir_figures.mkdir(parents=True, exist_ok=True)
    
    for held_out_battery in battery_ids:
        if held_out_battery not in df_with_targets["battery_id"].unique():
            continue
            
        train_df, test_df = leave_one_battery_out(df_with_targets, held_out_battery)
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        # We need a dataframe to align predictions
        # We construct it step by step from test_df to ensure alignment
        preds_df = test_df[["battery_id", "discharge_cycle_index", target_col]].copy()
        preds_df.rename(columns={target_col: "actual_next_capacity"}, inplace=True)
        
        base_preds_list = []
        for name, model in base_models.items():
            cloned_model = clone(model)
            cloned_model.fit(X_train, y_train)
            tmp_df = test_df[["battery_id", "discharge_cycle_index"]].copy()
            tmp_df[name] = cloned_model.predict(X_test)
            base_preds_list.append(tmp_df)
            
        merged_preds = base_preds_list[0]
        for tmp_df in base_preds_list[1:]:
            merged_preds = pd.merge(merged_preds, tmp_df, on=["battery_id", "discharge_cycle_index"])
            
        preds_df = pd.merge(preds_df, merged_preds, on=["battery_id", "discharge_cycle_index"])
        
        for name, model in sklearn_ensembles.items():
            cloned_model = clone(model)
            cloned_model.fit(X_train, y_train)
            tmp_df = test_df[["battery_id", "discharge_cycle_index"]].copy()
            tmp_df[name] = cloned_model.predict(X_test)
            preds_df = pd.merge(preds_df, tmp_df, on=["battery_id", "discharge_cycle_index"])
            
        base_names = list(base_models.keys())
        
        preds_df["SimpleAveraging"] = preds_df[base_names].mean(axis=1)
        preds_df["WeightedAveraging"] = sum(preds_df[name] * weights[name] for name in base_names)
        
        all_model_names = base_names + list(sklearn_ensembles.keys()) + ["SimpleAveraging", "WeightedAveraging"]
        
        for model_name in all_model_names:
            y_pred = preds_df[model_name]
            
            mae = mean_absolute_error(preds_df["actual_next_capacity"], y_pred)
            rmse = mean_squared_error(preds_df["actual_next_capacity"], y_pred, squared=False)
            r2 = r2_score(preds_df["actual_next_capacity"], y_pred)
            
            if model_name in base_models:
                cfg = "Base Model"
            elif model_name in sklearn_ensembles:
                try:
                    cfg = str(sklearn_ensembles[model_name].get_params())
                except:
                    cfg = "Sklearn Ensemble"
            elif model_name == "SimpleAveraging":
                cfg = "Mean of base models"
            else:
                cfg = str(weights)
                
            results.append({
                "model_or_ensemble": model_name,
                "held_out_battery": held_out_battery,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "n_test_rows": len(preds_df),
                "configuration": cfg
            })
            
            for _, row in preds_df.iterrows():
                predictions_log.append({
                    "model_or_ensemble": model_name,
                    "battery_id": row["battery_id"],
                    "discharge_cycle_index": row["discharge_cycle_index"],
                    "actual_next_capacity": row["actual_next_capacity"],
                    "predicted_next_capacity": row[model_name]
                })
                
        # Error correlation
        residuals_df = pd.DataFrame()
        for name in base_names:
            # Re-assert alignment using battery_id and discharge_cycle_index
            # Since residuals_df is just built from columns of preds_df, they are aligned.
            # But the user asked for explicit alignment.
            # We will use the aligned preds_df to calculate residuals.
            residuals_df[name] = preds_df["actual_next_capacity"] - preds_df[name]
            
        corr_matrix = residuals_df.corr(method="pearson")
        corr_csv_path = out_dir_tables / f"residual_corr_{held_out_battery}.csv"
        corr_matrix.to_csv(corr_csv_path)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, fmt=".3f")
        plt.title(f"Residual Correlation - {held_out_battery}")
        plt.tight_layout()
        plt.savefig(out_dir_figures / f"residual_corr_{held_out_battery}.png")
        plt.close()

    df_results = pd.DataFrame(results)
    df_preds = pd.DataFrame(predictions_log)
    
    df_results.to_csv(out_dir_tables / "ensemble_lobo_results.csv", index=False)
    df_preds.to_csv(out_dir_tables / "ensemble_lobo_predictions.csv", index=False)
    print("Phase 6 experiment complete. Results saved.")

if __name__ == "__main__":
    main()
