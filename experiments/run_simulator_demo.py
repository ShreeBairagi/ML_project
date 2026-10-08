import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.battery.simulator import simulate_battery_population

def main():
    # Fixed seed for demonstration
    random_state = 12345
    
    # Generate a population of 4 synthetic batteries
    print(f"Generating synthetic battery population with fixed random_state={random_state}...")
    df = simulate_battery_population(n_batteries=4, n_cycles=170, random_state=random_state)
    
    # Ensure output directories exist
    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the synthetic data
    csv_path = tables_dir / "synthetic_battery_population.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic data to: {csv_path}")
    
    # Create sanity-check figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    battery_ids = df["battery_id"].unique()
    for i, bat_id in enumerate(battery_ids):
        ax = axes[i]
        bat_df = df[df["battery_id"] == bat_id]
        
        ax.scatter(bat_df["discharge_cycle_index"], bat_df["capacity"], 
                   color='blue', alpha=0.5, s=10, label='Observed Capacity')
        ax.plot(bat_df["discharge_cycle_index"], bat_df["true_capacity"], 
                color='red', linewidth=2, label='True Capacity')
        
        ax.set_title(f"Battery: {bat_id}")
        ax.set_xlabel("Discharge Cycle Index")
        ax.set_ylabel("Capacity (Ah)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    figure_path = figures_dir / "synthetic_capacity_curves.png"
    plt.savefig(figure_path)
    plt.close()
    print(f"Saved sanity-check figure to: {figure_path}")
    print("\n--- Summary Statistics ---")
    
    # Print summary output per battery
    for bat_id in battery_ids:
        bat_df = df[df["battery_id"] == bat_id]
        n_cycles = len(bat_df)
        
        # Initial capacities (at index 0)
        initial_obs = bat_df.iloc[0]["capacity"]
        initial_true = bat_df.iloc[0]["true_capacity"]
        
        # Final capacities (at last index)
        final_obs = bat_df.iloc[-1]["capacity"]
        final_true = bat_df.iloc[-1]["true_capacity"]
        
        # Min/Max observed capacity
        min_obs = bat_df["capacity"].min()
        max_obs = bat_df["capacity"].max()
        
        print(f"\nBattery {bat_id}:")
        print(f"  Number of cycles:          {n_cycles}")
        print(f"  Initial observed capacity: {initial_obs:.4f} Ah")
        print(f"  Final observed capacity:   {final_obs:.4f} Ah")
        print(f"  Initial true capacity:     {initial_true:.4f} Ah")
        print(f"  Final true capacity:       {final_true:.4f} Ah")
        print(f"  Minimum observed capacity: {min_obs:.4f} Ah")
        print(f"  Maximum observed capacity: {max_obs:.4f} Ah")

if __name__ == "__main__":
    main()
