import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.battery import loading
from src.battery import splits
from src.battery import targets

def get_prepared_data(data_dir: str = "data/raw"):
    battery_ids = ["B0005", "B0006", "B0007", "B0018"]
    all_dfs = []
    for bid in battery_ids:
        file_path = os.path.join(data_dir, f"{bid}.mat")
        ops = loading.load_battery_data(file_path, bid)
        df_summary = loading.create_summary_dataframe(ops)
        
        # Keep only discharge
        df_discharge = df_summary[df_summary['operation_type'] == 'discharge'].copy()
        
        # Add target
        df_target = targets.add_next_cycle_capacity_target(df_discharge)
        all_dfs.append(df_target)
        
    return pd.concat(all_dfs, ignore_index=True)

def run_experiment():
    os.makedirs('results/tables', exist_ok=True)
    df = get_prepared_data()
    
    features = ['discharge_cycle_index', 'capacity']
    # Explicitly allowed: current-cycle `capacity` is allowed because the primary task 
    # predicts capacity at cycle t+1, with the prediction made at the end of cycle t.
    
    target_col = 'target_next_capacity'
    alpha = 1.0
    
    # -------------------------------------------------------------------------
    # Experiment A: Split Leakage
    # -------------------------------------------------------------------------
    results_a = []
    
    # Condition A1: WRONG_RANDOM_CYCLE_SPLIT
    seeds_a1 = [0, 1, 2, 3, 4]
    for seed in seeds_a1:
        train_df, test_df = splits.random_cycle_split_wrong(df, random_state=seed)
        
        X_train, y_train = train_df[features], train_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        results_a.append({
            'experiment': 'A_Split_Leakage',
            'condition': 'WRONG_RANDOM_CYCLE_SPLIT',
            'split_type': 'random',
            'battery_id': 'all_mixed',
            'seed': seed,
            'alpha': alpha,
            'feature_set': str(features),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'n_test_rows': len(X_test)
        })
        
    # Condition A2: CORRECT_LOBO
    battery_ids = df['battery_id'].unique()
    for bid in battery_ids:
        train_df, test_df = splits.leave_one_battery_out(df, held_out_battery=bid)
        
        X_train, y_train = train_df[features], train_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]
        
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        results_a.append({
            'experiment': 'A_Split_Leakage',
            'condition': 'CORRECT_LOBO',
            'split_type': 'LOBO',
            'battery_id': bid,
            'seed': 'N/A',
            'alpha': alpha,
            'feature_set': str(features),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'n_test_rows': len(X_test)
        })
        
    pd.DataFrame(results_a).to_csv('results/tables/ridge_split_leakage_results.csv', index=False)
    
    # -------------------------------------------------------------------------
    # Experiment B: Preprocessing Leakage
    # -------------------------------------------------------------------------
    results_b = []
    
    # We use random split seed 42 to keep the split identical for both conditions
    train_df, test_df = splits.random_cycle_split_wrong(df, random_state=42)
    
    # Condition B1: CORRECT_SCALER
    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]
    
    pipe_correct = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    
    pipe_correct.fit(X_train, y_train)
    y_pred_correct = pipe_correct.predict(X_test)
    
    results_b.append({
        'experiment': 'B_Preprocessing_Leakage',
        'condition': 'CORRECT_SCALER',
        'split_type': 'random',
        'battery_id': 'all_mixed',
        'seed': 42,
        'alpha': alpha,
        'feature_set': str(features),
        'MAE': mean_absolute_error(y_test, y_pred_correct),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_correct)),
        'R2': r2_score(y_test, y_pred_correct),
        'n_test_rows': len(X_test)
    })
    
    # Condition B2: WRONG_GLOBAL_SCALER
    # Deliberately wrong: fit scaler on all X, then split using same indices
    # We can do this by applying the global scaler to the whole dataframe first
    global_scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = global_scaler.fit_transform(df[features])
    
    # Now use the SAME split logic so we get exactly the same indices in train and test
    train_scaled, test_scaled = splits.random_cycle_split_wrong(df_scaled, random_state=42)
    
    X_train_wrong = train_scaled[features]
    y_train_wrong = train_scaled[target_col]
    X_test_wrong = test_scaled[features]
    y_test_wrong = test_scaled[target_col]
    
    ridge_wrong = Ridge(alpha=alpha)
    ridge_wrong.fit(X_train_wrong, y_train_wrong)
    y_pred_wrong = ridge_wrong.predict(X_test_wrong)
    
    results_b.append({
        'experiment': 'B_Preprocessing_Leakage',
        'condition': 'WRONG_GLOBAL_SCALER',
        'split_type': 'random',
        'battery_id': 'all_mixed',
        'seed': 42,
        'alpha': alpha,
        'feature_set': str(features),
        'MAE': mean_absolute_error(y_test_wrong, y_pred_wrong),
        'RMSE': np.sqrt(mean_squared_error(y_test_wrong, y_pred_wrong)),
        'R2': r2_score(y_test_wrong, y_pred_wrong),
        'n_test_rows': len(X_test_wrong)
    })
    
    pd.DataFrame(results_b).to_csv('results/tables/ridge_preprocessing_leakage_results.csv', index=False)
    
    print("Experiment Phase 3B complete.")
    print("Files written:")
    print(" - results/tables/ridge_split_leakage_results.csv")
    print(" - results/tables/ridge_preprocessing_leakage_results.csv")

if __name__ == "__main__":
    run_experiment()
