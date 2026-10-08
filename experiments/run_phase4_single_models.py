import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.battery.loading import load_battery_data, create_summary_dataframe
from src.battery.targets import add_next_cycle_capacity_target
from src.battery.splits import leave_one_battery_out
from src.battery.features import get_feature_columns
from src.battery.models import get_phase4_models

def main():
    data_dir = project_root / "data" / "raw"
    battery_ids = ["B0005", "B0006", "B0007", "B0018"]
    
    # 1. Load data
    all_summaries = []
    for bid in battery_ids:
        file_path = data_dir / f"{bid}.mat"
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping {bid}.")
            continue
        ops = load_battery_data(str(file_path), bid)
        df_summary = create_summary_dataframe(ops)
        
        # Keep only discharge cycles with valid capacity
        df_discharge = df_summary[(df_summary["operation_type"] == "discharge") & (df_summary["capacity"].notna())].copy()
        all_summaries.append(df_discharge)
        
    if not all_summaries:
        print("No data loaded. Exiting.")
        return

    df_all = pd.concat(all_summaries, ignore_index=True)
    
    # 2. Add targets
    df_with_targets = add_next_cycle_capacity_target(df_all)
    
    features = get_feature_columns()
    target_col = "target_next_capacity"
    
    # Assert no future features
    assert target_col not in features, "target_next_capacity must not be in features!"
    
    models = get_phase4_models()
    
    results = []
    predictions_log = []
    
    # 3. Evaluate using LOBO
    for held_out_battery in battery_ids:
        if held_out_battery not in df_with_targets["battery_id"].unique():
            continue
            
        train_df, test_df = leave_one_battery_out(df_with_targets, held_out_battery)
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            
            model_params = str(model.get_params())
            
            results.append({
                "model": model_name,
                "held_out_battery": held_out_battery,
                "feature_set": str(features),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "n_test_rows": len(y_test),
                "model_parameters": model_params
            })
            
            for i, (idx, row) in enumerate(test_df.iterrows()):
                predictions_log.append({
                    "model": model_name,
                    "battery_id": held_out_battery,
                    "discharge_cycle_index": row["discharge_cycle_index"],
                    "actual_next_capacity": y_test.iloc[i],
                    "predicted_next_capacity": y_pred[i]
                })

    df_results = pd.DataFrame(results)
    df_preds = pd.DataFrame(predictions_log)
    
    out_dir = project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_results.to_csv(out_dir / "single_model_lobo_results.csv", index=False)
    df_preds.to_csv(out_dir / "single_model_lobo_predictions.csv", index=False)
    
    print("Experiment completed.")
    print("Files saved: ")
    print(f" - {out_dir / 'single_model_lobo_results.csv'}")
    print(f" - {out_dir / 'single_model_lobo_predictions.csv'}")

if __name__ == "__main__":
    main()
