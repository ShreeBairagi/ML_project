import os
import pytest
from src.battery.loading import load_battery_data, create_summary_dataframe

RAW_DATA_DIR = "data/raw"
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

def get_battery_file(battery_id: str) -> str:
    return os.path.join(RAW_DATA_DIR, f"{battery_id}.mat")

def check_data_availability():
    for b_id in BATTERIES:
        if not os.path.exists(get_battery_file(b_id)):
            pytest.skip(f"Raw data file for {b_id} missing. Skipping loading tests.")

@pytest.fixture(scope="module")
def all_loaded_data():
    check_data_availability()
    data = {}
    for b_id in BATTERIES:
        data[b_id] = load_battery_data(get_battery_file(b_id), b_id)
    return data

def test_files_load_successfully(all_loaded_data):
    for b_id in BATTERIES:
        assert b_id in all_loaded_data
        assert len(all_loaded_data[b_id]) > 0

def test_operation_index_ordered(all_loaded_data):
    for b_id, ops in all_loaded_data.items():
        indices = [op['operation_index'] for op in ops]
        assert indices == list(range(len(ops)))

def test_discharge_cycle_index(all_loaded_data):
    for b_id, ops in all_loaded_data.items():
        discharge_indices = [op['discharge_cycle_index'] for op in ops if op['operation_type'] == 'discharge']
        # Check consecutive
        assert discharge_indices == list(range(len(discharge_indices)))
        
        # Check None for non-discharge
        non_discharge_indices = [op['discharge_cycle_index'] for op in ops if op['operation_type'] != 'discharge']
        assert all(idx is None for idx in non_discharge_indices)

def test_unique_battery_operation_index(all_loaded_data):
    seen = set()
    for b_id, ops in all_loaded_data.items():
        for op in ops:
            identifier = (b_id, op['operation_index'])
            assert identifier not in seen
            seen.add(identifier)

def test_capacity_presence(all_loaded_data):
    for b_id, ops in all_loaded_data.items():
        for op in ops:
            if op['operation_type'] == 'discharge':
                assert op['capacity'] is not None
                assert isinstance(op['capacity'], float)
            else:
                assert op['capacity'] is None

def test_summary_dataframe(all_loaded_data):
    b_id = "B0005"
    ops = all_loaded_data[b_id]
    df = create_summary_dataframe(ops)
    
    assert len(df) == len(ops)
    assert list(df.columns) == [
        'battery_id', 'operation_index', 'operation_type', 
        'discharge_cycle_index', 'ambient_temperature', 'timestamp', 'capacity'
    ]

def test_b0005_has_discharge(all_loaded_data):
    ops = all_loaded_data["B0005"]
    discharge_ops = [op for op in ops if op['operation_type'] == 'discharge']
    assert len(discharge_ops) > 0

def test_valid_operation_types(all_loaded_data):
    for b_id, ops in all_loaded_data.items():
        first_op_type = ops[0]['operation_type']
        assert first_op_type in {'charge', 'discharge', 'impedance'}

def test_discharge_count_greater_than_zero(all_loaded_data):
    for b_id, ops in all_loaded_data.items():
        discharge_ops = [op for op in ops if op['operation_type'] == 'discharge']
        assert len(discharge_ops) > 0

def test_discharge_row_has_numeric_capacity(all_loaded_data):
    for b_id, ops in all_loaded_data.items():
        for op in ops:
            if op['operation_type'] == 'discharge':
                assert isinstance(op['capacity'], float)

def test_raw_measurement_shape_preservation(all_loaded_data):
    import numpy as np
    ops = all_loaded_data["B0005"]
    discharge_ops = [op for op in ops if op['operation_type'] == 'discharge']
    
    assert len(discharge_ops) > 0
    first_discharge = discharge_ops[0]
    
    assert 'Voltage_measured' in first_discharge
    voltage = first_discharge['Voltage_measured']
    assert isinstance(voltage, np.ndarray)
    assert len(voltage.shape) >= 2

def test_malformed_scalar_extraction():
    from src.battery.loading import _extract_scalar, _extract_string, _extract_time
    import numpy as np
    
    with pytest.raises(ValueError):
        _extract_scalar(np.array([1, 2]))
        
    with pytest.raises(ValueError):
        _extract_string(np.array([]))
        
    with pytest.raises(ValueError):
        _extract_time(np.array([1, 2, 3]))
