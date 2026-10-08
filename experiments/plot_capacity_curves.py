import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure src module is in path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.battery.loading import load_battery_data, create_summary_dataframe

def main():
    data_dir = project_root / "data" / "raw"
    results_dir = project_root / "results" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    batteries = ['B0005', 'B0006', 'B0007', 'B0018']
    
    # Setup subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    stats = []
    increase_counts = {}
    
    for i, bat_id in enumerate(batteries):
        file_path = data_dir / f"{bat_id}.mat"
        operations = load_battery_data(str(file_path), bat_id)
        df_all = create_summary_dataframe(operations)
        
        # Extract only discharge operations
        df_discharge = df_all[df_all['operation_type'] == 'discharge'].copy()
        
        # Sort by discharge_cycle_index just in case
        df_discharge = df_discharge.sort_values('discharge_cycle_index').reset_index(drop=True)
        
        cycles = df_discharge['discharge_cycle_index'].values
        capacities = df_discharge['capacity'].values
        
        num_cycles = len(cycles)
        if num_cycles == 0:
            raise ValueError(f"Battery {bat_id} has zero discharge cycles.")
            
        # 1. Descriptive stats
        first_cap = capacities[0]
        last_cap = capacities[-1]
        min_cap = np.min(capacities)
        max_cap = np.max(capacities)
        min_idx = cycles[np.argmin(capacities)]
        max_idx = cycles[np.argmax(capacities)]
        
        stats.append({
            'Battery': bat_id,
            'Discharge Cycles': num_cycles,
            'First Capacity': first_cap,
            'Last Capacity': last_cap,
            'Min Capacity': min_cap,
            'Max Capacity': max_cap,
            'Cycle of Min': min_idx,
            'Cycle of Max': max_idx
        })
        
        # 2. Data quality checks
        
        # Check finite capacity
        if not np.all(np.isfinite(capacities)):
            print(f"WARNING: {bat_id} contains non-finite capacity values!")
            
        # Check no duplicates
        if len(set(cycles)) != len(cycles):
            print(f"WARNING: {bat_id} contains duplicate discharge_cycle_index values!")
            
        # Check monotonic and consecutive
        expected_cycles = np.arange(cycles[0], cycles[0] + len(cycles))
        if not np.array_equal(cycles, expected_cycles):
            print(f"WARNING: {bat_id} discharge_cycle_index is not monotonic and consecutive!")
            
        # Count capacity increases between adjacent cycles
        diffs = np.diff(capacities)
        increases = np.sum(diffs > 0)
        increase_counts[bat_id] = int(increases)
        
        # 3. Plotting
        ax = axes[i]
        ax.plot(cycles, capacities, marker='.', linestyle='-', alpha=0.7)
        ax.set_title(f"Battery {bat_id} Capacity vs Cycle")
        ax.set_xlabel("Discharge Cycle Index")
        ax.set_ylabel("Capacity (Ah)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
    # Save figure
    plt.tight_layout()
    fig_path = results_dir / "capacity_vs_cycle.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    # Print descriptive table
    stats_df = pd.DataFrame(stats)
    print("\n--- Descriptive Statistics Table ---")
    print(stats_df.to_string(index=False))
    
    # Print increase counts
    print("\n--- Adjacent Capacity Increases ---")
    for bat, count in increase_counts.items():
        print(f"{bat}: {count} increases")
        
    print(f"\nFigure saved to: {fig_path.relative_to(project_root)}")
    
if __name__ == "__main__":
    main()
