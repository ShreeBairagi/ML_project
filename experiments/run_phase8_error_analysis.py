import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.battery.loading import load_battery_data, create_summary_dataframe
from src.battery.targets import add_next_cycle_capacity_target
from src.battery.splits import leave_one_battery_out
from src.battery.features import get_feature_columns
from src.battery.simulator import simulate_battery_population, simulate_battery_capacity

def get_phase8_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])

def assign_life_regions(df):
    """Assign early, middle, late based on deterministic thirds of row position."""
    n = len(df)
    if n == 0:
        return []
    third1 = n // 3
    third2 = 2 * n // 3
    regions = ['early'] * third1 + ['middle'] * (third2 - third1) + ['late'] * (n - third2)
    return regions

def run_part_a(df_with_targets, battery_ids, features, target_col):
    out_dir_tables = project_root / "results" / "tables"
    out_dir_figures = project_root / "results" / "figures"
    out_dir_tables.mkdir(parents=True, exist_ok=True)
    out_dir_figures.mkdir(parents=True, exist_ok=True)
    
    region_metrics = []
    all_predictions = []
    largest_errors = []
    
    for held_out_battery in battery_ids:
        if held_out_battery not in df_with_targets["battery_id"].unique():
            continue
            
        train_df, test_df = leave_one_battery_out(df_with_targets, held_out_battery)
        
        # Ensure chronological order
        test_df = test_df.sort_values("discharge_cycle_index").reset_index(drop=True)
        test_df["life_region"] = assign_life_regions(test_df)
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        
        X_test = test_df[features]
        y_test = test_df[target_col]
        
        pipeline = get_phase8_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        test_df["actual_next_capacity"] = y_test.values
        test_df["predicted_next_capacity"] = y_pred
        test_df["residual"] = test_df["actual_next_capacity"] - test_df["predicted_next_capacity"]
        test_df["absolute_error"] = test_df["residual"].abs()
        
        # Region metrics
        for region, group in test_df.groupby("life_region"):
            region_metrics.append({
                "battery_id": held_out_battery,
                "life_region": region,
                "n_rows": len(group),
                "MAE": mean_absolute_error(group["actual_next_capacity"], group["predicted_next_capacity"]),
                "RMSE": mean_squared_error(group["actual_next_capacity"], group["predicted_next_capacity"], squared=False),
                "mean_residual": group["residual"].mean(),
                "median_absolute_error": group["absolute_error"].median(),
                "maximum_absolute_error": group["absolute_error"].max()
            })
            
        all_predictions.append(test_df)
        
        # Top 10 worst predictions
        top_10 = test_df.nlargest(10, "absolute_error").copy()
        largest_errors.append(top_10[[
            "battery_id", "discharge_cycle_index", "actual_next_capacity", 
            "predicted_next_capacity", "residual", "absolute_error", "life_region"
        ]])
        
        # Plots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(test_df["discharge_cycle_index"], test_df["absolute_error"], marker='o', linestyle='-', color='red')
        axes[0].set_ylabel("Absolute Error")
        axes[0].set_title(f"Prediction Errors over Life - {held_out_battery}")
        axes[0].grid(True, alpha=0.5)
        
        axes[1].plot(test_df["discharge_cycle_index"], test_df["residual"], marker='o', linestyle='-', color='blue')
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].set_xlabel("Discharge Cycle Index")
        axes[1].set_ylabel("Residual (Actual - Predicted)")
        axes[1].grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(out_dir_figures / f"error_over_cycle_{held_out_battery}.png")
        plt.close()
        
    df_region_metrics = pd.DataFrame(region_metrics)
    df_region_metrics.to_csv(out_dir_tables / "error_by_life_region.csv", index=False)
    
    df_all_predictions = pd.concat(all_predictions, ignore_index=True)
    df_all_predictions.to_csv(out_dir_tables / "error_analysis_predictions.csv", index=False)
    
    df_largest_errors = pd.concat(largest_errors, ignore_index=True)
    df_largest_errors.to_csv(out_dir_tables / "largest_errors.csv", index=False)
    
    return df_region_metrics, df_all_predictions, df_largest_errors

def run_part_b(features, target_col):
    out_dir_tables = project_root / "results" / "tables"
    out_dir_tables.mkdir(parents=True, exist_ok=True)
    
    # 1. Training population
    df_train_raw = simulate_battery_population(n_batteries=8, n_cycles=170, random_state=42)
    # The simulate_battery_population creates SYN01 to SYN08
    
    # 2. In-distribution test population
    df_test_id_raw = simulate_battery_population(n_batteries=4, n_cycles=170, random_state=100)
    # Change names to ensure disjoint
    df_test_id_raw["battery_id"] = df_test_id_raw["battery_id"].str.replace("SYN", "ID_SYN")
    
    # 3. Shifted test population
    shifted_frames = []
    rng = np.random.default_rng(200)
    
    # We record exactly what shift we apply
    shift_params = []
    
    for i in range(4):
        # We enforce a stronger degradation shift
        initial_capacity = rng.uniform(1.6, 1.8)
        linear_fade = rng.uniform(0.0035, 0.0050)
        quadratic_fade = rng.uniform(0.000010, 0.000020)
        
        shift_params.append({
            "battery_id": f"SHIFT_{i+1:02d}",
            "initial_capacity": initial_capacity,
            "linear_fade": linear_fade,
            "quadratic_fade": quadratic_fade
        })
        
        frame = simulate_battery_capacity(
            battery_id=f"SHIFT_{i+1:02d}",
            n_cycles=170,
            initial_capacity=initial_capacity,
            linear_fade=linear_fade,
            quadratic_fade=quadratic_fade,
            noise_std=0.01,
            random_state=200 + i
        )
        shifted_frames.append(frame)
        
    df_test_ood_raw = pd.concat(shifted_frames, ignore_index=True)
    
    # Add targets
    df_train = add_next_cycle_capacity_target(df_train_raw)
    df_test_id = add_next_cycle_capacity_target(df_test_id_raw)
    df_test_ood = add_next_cycle_capacity_target(df_test_ood_raw)
    
    X_train = df_train[features]
    y_train = df_train[target_col]
    
    X_id = df_test_id[features]
    y_id = df_test_id[target_col]
    
    X_ood = df_test_ood[features]
    y_ood = df_test_ood[target_col]
    
    pipeline = get_phase8_pipeline()
    pipeline.fit(X_train, y_train)
    
    y_pred_id = pipeline.predict(X_id)
    y_pred_ood = pipeline.predict(X_ood)
    
    metrics = {
        "condition": ["in_distribution", "shifted"],
        "MAE": [mean_absolute_error(y_id, y_pred_id), mean_absolute_error(y_ood, y_pred_ood)],
        "RMSE": [mean_squared_error(y_id, y_pred_id, squared=False), mean_squared_error(y_ood, y_pred_ood, squared=False)],
        "R2": [r2_score(y_id, y_pred_id), r2_score(y_ood, y_pred_ood)],
        "n_test_rows": [len(y_id), len(y_ood)]
    }
    
    df_metrics = pd.DataFrame(metrics)
    
    # Calculate differences
    shift_diff = {
        "condition": "difference (shifted - in_distribution)",
        "MAE": metrics["MAE"][1] - metrics["MAE"][0],
        "RMSE": metrics["RMSE"][1] - metrics["RMSE"][0],
        "R2": metrics["R2"][1] - metrics["R2"][0],
        "n_test_rows": None
    }
    
    df_metrics = pd.concat([df_metrics, pd.DataFrame([shift_diff])], ignore_index=True)
    df_metrics.to_csv(out_dir_tables / "synthetic_shift_results.csv", index=False)
    
    # Save predictions optionally
    df_test_id_preds = df_test_id.copy()
    df_test_id_preds["condition"] = "in_distribution"
    df_test_id_preds["predicted_next_capacity"] = y_pred_id
    
    df_test_ood_preds = df_test_ood.copy()
    df_test_ood_preds["condition"] = "shifted"
    df_test_ood_preds["predicted_next_capacity"] = y_pred_ood
    
    df_all_syn_preds = pd.concat([df_test_id_preds, df_test_ood_preds], ignore_index=True)
    df_all_syn_preds.to_csv(out_dir_tables / "synthetic_shift_predictions.csv", index=False)
    
    return shift_params, df_train, df_test_id, df_test_ood, df_metrics

def main():
    data_dir = project_root / "data" / "raw"
    battery_ids = ["B0005", "B0006", "B0007", "B0018"]
    
    # Load data for Part A
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
    df_with_targets = add_next_cycle_capacity_target(df_all)
    
    features = get_feature_columns()
    target_col = "target_next_capacity"
    
    assert target_col not in features, "target_next_capacity must not be in features!"
    
    run_part_a(df_with_targets, battery_ids, features, target_col)
    shift_params, _, _, _, _ = run_part_b(features, target_col)
    
    # Print the required output block
    print("\n--- PHASE 8 COMPLETED ---")
    print("\nExact Ridge Configuration:")
    print("Pipeline(steps=[('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])")
    
    print("\nExact life-region definition:")
    print("Deterministic thirds based on row position/order (sorted by cycle index).")
    print("early = first n//3 rows, middle = next (2*n)//3 - n//3 rows, late = remaining rows.")
    
    print("\nBattery x Region Error Metrics:")
    df_region_metrics = pd.read_csv(project_root / "results" / "tables" / "error_by_life_region.csv")
    print(df_region_metrics.to_string(index=False))
    
    print("\nLargest Error Cases (top 10 per battery):")
    df_largest_errors = pd.read_csv(project_root / "results" / "tables" / "largest_errors.csv")
    print(df_largest_errors.to_string(index=False))
    
    print("\nExact Synthetic Train/ID/Shift parameters:")
    print("Train: n_batteries=8, n_cycles=170, random_state=42 (default ranges: IC=1.9-2.1, LF=0.0018-0.0032, QF=2e-6-8e-6)")
    print("In-Distribution (ID): n_batteries=4, n_cycles=170, random_state=100 (same default ranges)")
    print("Shifted (OOD): 4 batteries, 170 cycles, controlled shift parameters:")
    for sp in shift_params:
        print(f"  {sp['battery_id']}: IC={sp['initial_capacity']:.4f}, LF={sp['linear_fade']:.6f}, QF={sp['quadratic_fade']:.6f}")
    
    print("\nSynthetic Shift Metrics:")
    df_shift = pd.read_csv(project_root / "results" / "tables" / "synthetic_shift_results.csv")
    print(df_shift.to_string(index=False))

    print("\nSaved Paths:")
    print(f"  {project_root / 'results' / 'tables' / 'error_by_life_region.csv'}")
    print(f"  {project_root / 'results' / 'tables' / 'error_analysis_predictions.csv'}")
    print(f"  {project_root / 'results' / 'tables' / 'largest_errors.csv'}")
    print(f"  {project_root / 'results' / 'figures' / 'error_over_cycle_B0005.png'} (and others)")
    print(f"  {project_root / 'results' / 'tables' / 'synthetic_shift_results.csv'}")
    print(f"  {project_root / 'results' / 'tables' / 'synthetic_shift_predictions.csv'}")

if __name__ == "__main__":
    main()
