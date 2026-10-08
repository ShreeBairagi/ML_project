import numpy as np
import pandas as pd
import scipy.io

def _extract_string(val) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, np.ndarray) and val.size > 0:
        return str(val.flat[0])
    raise ValueError(f"Cannot extract string from malformed structure: {val}")

def _extract_scalar(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, np.ndarray) and val.size == 1:
        return float(val.item())
    raise ValueError(f"Cannot extract scalar from malformed structure of shape {getattr(val, 'shape', None)}")

def _extract_time(val) -> str:
    if isinstance(val, np.ndarray):
        flat_val = val.flatten()
        if flat_val.size == 6:
            return f"{int(flat_val[0])}-{int(flat_val[1]):02d}-{int(flat_val[2]):02d} {int(flat_val[3]):02d}:{int(flat_val[4]):02d}:{int(flat_val[5]):02d}"
    raise ValueError(f"Malformed time array: {val}")

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
        op_type = _extract_string(op['type'])
        ambient_temp = _extract_scalar(op['ambient_temperature'])
        timestamp = _extract_time(op['time'])
            
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
                op_dict['capacity'] = _extract_scalar(data_struct['Capacity'])
            discharge_cycle_index += 1
            
        # Dynamically retain all actual confirmed nested data fields for each operation
        for field in data_fields:
            if field != 'Capacity':
                op_dict[field] = data_struct[field]
                
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
