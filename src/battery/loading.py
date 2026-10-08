import numpy as np
import pandas as pd
import scipy.io

def load_battery_data(file_path: str, battery_id: str) -> list[dict]:
    """
    Loads battery data from a NASA .mat file into a list of operation dictionaries.
    Maintains raw nested data arrays and extracts scalar metadata.
    """
    mat = scipy.io.loadmat(file_path)
    battery_struct = mat[battery_id]
    cycles = battery_struct[0, 0]['cycle'][0]
    
    operations = []
    discharge_cycle_index = 0
    
    for op_idx, op in enumerate(cycles):
        op_type = str(op['type'][0])
        
        try:
            ambient_temp = float(op['ambient_temperature'][0, 0])
        except (IndexError, ValueError, TypeError):
            ambient_temp = float(op['ambient_temperature'][0]) if len(op['ambient_temperature']) > 0 else np.nan
            
        time_array = op['time'][0]
        try:
            timestamp = f"{int(time_array[0])}-{int(time_array[1]):02d}-{int(time_array[2]):02d} {int(time_array[3]):02d}:{int(time_array[4]):02d}:{int(time_array[5]):02d}"
        except Exception:
            timestamp = str(time_array)
            
        data_struct = op['data'][0, 0]
        data_fields = data_struct.dtype.names
        
        op_dict = {
            'battery_id': battery_id,
            'operation_index': op_idx,
            'operation_type': op_type,
            'ambient_temperature': ambient_temp,
            'timestamp': timestamp,
            'discharge_cycle_index': None,
            'capacity': None
        }
        
        if op_type == 'discharge':
            op_dict['discharge_cycle_index'] = discharge_cycle_index
            if 'Capacity' in data_fields:
                op_dict['capacity'] = float(data_struct['Capacity'][0, 0])
            discharge_cycle_index += 1
            
        # Dynamically retain all actual confirmed nested data fields for each operation
        for field in data_fields:
            if field != 'Capacity':
                op_dict[field] = data_struct[field].flatten()
                
        operations.append(op_dict)
        
    return operations

def create_summary_dataframe(operations: list[dict]) -> pd.DataFrame:
    """
    Creates a tidy summary DataFrame containing only scalar metadata.
    Explicitly excludes full measurement arrays.
    """
    scalar_keys = [
        'battery_id', 'operation_index', 'operation_type', 
        'discharge_cycle_index', 'ambient_temperature', 'timestamp', 'capacity'
    ]
    
    records = [{k: op[k] for k in scalar_keys} for op in operations]
    return pd.DataFrame(records)
