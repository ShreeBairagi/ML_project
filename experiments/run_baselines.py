import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.battery.loading import load_battery_data, create_summary_dataframe
from src.battery.targets import add_next_cycle_capacity_target
from src.battery.splits import leave_one_battery_out, random_cycle_split_wrong, time_ordered_within_battery

def get_last_observed_value(prefix_df: pd.DataFrame) -> float:
    return prefix_df['capacity'].iloc[-1]

def exp_func(x, a, b, c):
    # Using np.errstate to handle overflow during curve fitting
    with np.errstate(over='ignore', invalid='ignore'):
        return a * np.exp(b * x) + c

def get_exponential_baseline(prefix_df: pd.DataFrame, target_cycle: int):
    if len(prefix_df) < 3:
        return get_last_observed_value(prefix_df), 'fallback_last_observed'
    
    x = prefix_df['discharge_cycle_index'].values.astype(float)
    y = prefix_df['capacity'].values.astype(float)
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            # Try to fit with a reasonable guess
            popt, _ = scipy.optimize.curve_fit(
                exp_func, x, y, maxfev=10000, p0=[1.0, -0.001, 1.0]
            )
        pred = exp_func(float(target_cycle), *popt)
        if not np.isfinite(pred):
            return get_last_observed_value(prefix_df), 'fallback_last_observed'
        return pred, 'exponential'
    except (RuntimeError, ValueError, OptimizeWarning):
        return get_last_observed_value(prefix_df), 'fallback_last_observed'

def evaluate_baselines(df, test_df, split_name, seed=None):
    results = []
    
    for idx, test_row in test_df.iterrows():
        bat_id = test_row['battery_id']
        current_cycle = test_row['discharge_cycle_index']
        actual_next = test_row['target_next_capacity']
        target_cycle = current_cycle + 1
        
        # Prefix extraction: strictly up to current_cycle
        prefix_df = df[(df['battery_id'] == bat_id) & (df['discharge_cycle_index'] <= current_cycle)].copy()
        
        if prefix_df.empty:
            continue
            
        prefix_df = prefix_df.sort_values('discharge_cycle_index')
        
        # 1. Last Observed Value
        lov_pred = get_last_observed_value(prefix_df)
        results.append({
            'battery_id': bat_id,
            'discharge_cycle_index': current_cycle,
            'actual_next_capacity': actual_next,
            'predicted_next_capacity': lov_pred,
            'baseline': 'last_observed_value',
            'split_type': split_name,
            'seed': seed,
            'prediction_method': 'last_observed_value'
        })
        
        # 2. Exponential Fade
        exp_pred, method = get_exponential_baseline(prefix_df, target_cycle)
        results.append({
            'battery_id': bat_id,
            'discharge_cycle_index': current_cycle,
            'actual_next_capacity': actual_next,
            'predicted_next_capacity': exp_pred,
            'baseline': 'exponential',
            'split_type': split_name,
            'seed': seed,
            'prediction_method': method
        })
        
    return results

def calculate_metrics(results_df):
    metrics = []
    groups = results_df.groupby(['split_type', 'baseline', 'battery_id', 'seed'], dropna=False)
    for (split, base, bat, seed), group in groups:
        actual = group['actual_next_capacity']
        pred = group['predicted_next_capacity']
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        if len(actual) > 1 and actual.var() > 0:
            r2 = r2_score(actual, pred)
        else:
            r2 = np.nan
            
        metrics.append({
            'split_type': split,
            'baseline': base,
            'battery_id': bat,
            'seed': seed,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'count': len(actual)
        })
    return pd.DataFrame(metrics)

def main():
    data_dir = project_root / "data" / "raw"
    batteries = ['B0005', 'B0006', 'B0007', 'B0018']
    
    print("Loading data...")
    all_discharge = []
    for bat_id in batteries:
        file_path = data_dir / f"{bat_id}.mat"
        operations = load_battery_data(str(file_path), bat_id)
        df_all = create_summary_dataframe(operations)
        df_discharge = df_all[df_all['operation_type'] == 'discharge'].copy()
        all_discharge.append(df_discharge)
        
    df = pd.concat(all_discharge, ignore_index=True)
    df = add_next_cycle_capacity_target(df)
    
    all_results = []
    
    # 1. LOBO
    print("Running LOBO evaluation...")
    for bat_id in batteries:
        train_df, test_df = leave_one_battery_out(df, bat_id)
        results = evaluate_baselines(df, test_df, 'leave_one_battery_out')
        all_results.extend(results)
        
    # 2. Time-Ordered
    print("Running Time-Ordered evaluation...")
    train_df, test_df = time_ordered_within_battery(df)
    results = evaluate_baselines(df, test_df, 'time_ordered_within_battery')
    all_results.extend(results)
    
    # 3. Random Cycle
    print("Running Random Cycle (WRONG / LEAKAGE-DEMO SPLIT) evaluation...")
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        train_df, test_df = random_cycle_split_wrong(df, random_state=seed)
        results = evaluate_baselines(df, test_df, 'random_cycle_split_wrong', seed=seed)
        all_results.extend(results)
        
    results_df = pd.DataFrame(all_results)
    
    out_dir = project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    preds_path = out_dir / "baseline_predictions.csv"
    results_df.to_csv(preds_path, index=False)
    
    metrics_df = calculate_metrics(results_df)
    mets_path = out_dir / "baseline_results.csv"
    metrics_df.to_csv(mets_path, index=False)
    
    print("\n--- Summary ---")
    print(f"Total prediction rows: {len(results_df)}")
    
    exp_failures = len(results_df[
        (results_df['baseline'] == 'exponential') & 
        (results_df['prediction_method'] == 'fallback_last_observed')
    ])
    print(f"Failed exponential fits (fallback used): {exp_failures}")
    print(f"Results saved to {preds_path} and {mets_path}")

if __name__ == "__main__":
    main()
