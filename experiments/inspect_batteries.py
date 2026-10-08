import os
import sys

# Ensure src module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.battery.loading import load_battery_data, create_summary_dataframe

def main():
    batteries = ["B0005", "B0006", "B0007", "B0018"]
    raw_dir = "data/raw"
    
    print("Battery Data Inspection:")
    print("-" * 50)
    
    for b_id in batteries:
        file_path = os.path.join(raw_dir, f"{b_id}.mat")
        if not os.path.exists(file_path):
            print(f"Skipping {b_id}: {file_path} not found.")
            continue
            
        ops = load_battery_data(file_path, b_id)
        df = create_summary_dataframe(ops)
        
        total_ops = len(df)
        type_counts = df['operation_type'].value_counts()
        charge_count = type_counts.get('charge', 0)
        discharge_count = type_counts.get('discharge', 0)
        impedance_count = type_counts.get('impedance', 0)
        
        discharge_df = df[df['operation_type'] == 'discharge']
        
        print(f"Battery: {b_id}")
        print(f"  Total operations: {total_ops}")
        print(f"  Charge count: {charge_count}")
        print(f"  Discharge count: {discharge_count}")
        print(f"  Impedance count: {impedance_count}")
        
        if not discharge_df.empty:
            first_cap = discharge_df.iloc[0]['capacity']
            last_cap = discharge_df.iloc[-1]['capacity']
            min_cap = discharge_df['capacity'].min()
            max_cap = discharge_df['capacity'].max()
            print(f"  First discharge capacity: {first_cap:.4f} Ah")
            print(f"  Last discharge capacity: {last_cap:.4f} Ah")
            print(f"  Min discharge capacity: {min_cap:.4f} Ah")
            print(f"  Max discharge capacity: {max_cap:.4f} Ah")
        else:
            print("  No discharge operations found.")
            
        print("-" * 50)

if __name__ == "__main__":
    main()
