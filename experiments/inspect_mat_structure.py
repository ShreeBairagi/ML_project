import scipy.io
import numpy as np

def inspect_mat():
    mat_path = 'data/raw/B0005.mat'
    print(f"Loading {mat_path}...")
    mat = scipy.io.loadmat(mat_path)
    
    print("\n--- Top-level keys ---")
    for key, val in mat.items():
        print(f"Key: {key}, Type: {type(val)}")
    
    # We expect 'B0005' to be the main key, let's check what it has
    if 'B0005' in mat:
        b5 = mat['B0005']
        print("\n--- Structure of B0005 ---")
        print(f"Type: {type(b5)}")
        if isinstance(b5, np.ndarray):
            print(f"Shape: {b5.shape}")
            print(f"Dtype fields: {b5.dtype.names}")
            
            # The structure is usually 1x1 array containing a struct with 'cycle'
            cycle_data = b5[0, 0]['cycle']
            print("\n--- Structure of the cycle array ---")
            print(f"Type: {type(cycle_data)}")
            print(f"Shape: {cycle_data.shape}")
            print(f"Dtype fields: {cycle_data.dtype.names}")
            
            # Usually cycle_data is a 1xN array
            cycles = cycle_data[0]
            print(f"Total number of cycles/operations: {cycles.size}")
            
            # Let's find unique operation types and print their nested data structure
            if 'type' in cycle_data.dtype.names:
                op_types = {}
                for i, op in enumerate(cycles):
                    # op['type'] is usually an array of string or just a string nested
                    op_type_val = op['type']
                    if isinstance(op_type_val, np.ndarray) and op_type_val.size > 0:
                        op_type_str = str(op_type_val[0])
                    else:
                        op_type_str = str(op_type_val)
                    
                    if op_type_str not in op_types:
                        op_types[op_type_str] = i
                        
                print("\n--- Found Operation Types ---")
                for op_str, idx in op_types.items():
                    print(f"\nOperation type '{op_str}' (first observed at index {idx}):")
                    op = cycles[idx]
                    
                    # Print standard fields
                    for field in cycle_data.dtype.names:
                        if field != 'data':
                            val = op[field]
                            if isinstance(val, np.ndarray) and val.size > 0:
                                val = val[0]
                            print(f"  {field}: {val}")
                    
                    # Print data field structure
                    if 'data' in cycle_data.dtype.names:
                        data_struct = op['data']
                        print(f"  Nested 'data' field structure:")
                        print(f"    Type: {type(data_struct)}")
                        print(f"    Shape: {data_struct.shape}")
                        if isinstance(data_struct, np.ndarray) and data_struct.dtype.names:
                            print(f"    Measurement fields: {data_struct.dtype.names}")
                            if data_struct.size > 0:
                                data_item = data_struct[0, 0] if data_struct.ndim == 2 else data_struct[0]
                                for nested_field in data_struct.dtype.names:
                                    meas = data_item[nested_field]
                                    meas_shape = meas.shape if isinstance(meas, np.ndarray) else 'scalar'
                                    meas_type = type(meas)
                                    print(f"      {nested_field}: shape {meas_shape}, type {meas_type}")
                                    if isinstance(meas, np.ndarray) and meas.size > 0:
                                        print(f"        (Inner dtype: {meas.dtype})")
                        else:
                            print(f"    (No nested structured fields found)")
            else:
                print("No 'type' field found in cycle array.")

if __name__ == '__main__':
    inspect_mat()
