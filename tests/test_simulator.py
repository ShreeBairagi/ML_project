import pytest
import pandas as pd
import numpy as np
from src.battery.simulator import simulate_battery_capacity, simulate_battery_population

def test_simulate_battery_capacity_rows_and_columns():
    n_cycles = 50
    df = simulate_battery_capacity("B01", n_cycles=n_cycles)
    
    # Returns requested number of rows
    assert len(df) == n_cycles
    
    # Required columns exist
    expected_cols = ["battery_id", "discharge_cycle_index", "capacity", "true_capacity"]
    for col in expected_cols:
        assert col in df.columns

def test_discharge_cycle_index_sequential():
    df = simulate_battery_capacity("B01", n_cycles=10)
    
    # starts at 0 and increases sequentially
    expected_index = np.arange(10)
    np.testing.assert_array_equal(df["discharge_cycle_index"].values, expected_index)

def test_random_state_reproducibility():
    df1 = simulate_battery_capacity("B01", random_state=42)
    df2 = simulate_battery_capacity("B01", random_state=42)
    
    # same random_state produces identical output
    pd.testing.assert_frame_equal(df1, df2)

def test_different_random_state_noise():
    df1 = simulate_battery_capacity("B01", random_state=42)
    df2 = simulate_battery_capacity("B01", random_state=100)
    
    # different random_state values produce different noisy observations
    # true_capacity should be the same, but capacity (observed) should differ
    pd.testing.assert_series_equal(df1["true_capacity"], df2["true_capacity"])
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(df1["capacity"], df2["capacity"])

def test_noise_std_zero():
    df = simulate_battery_capacity("B01", noise_std=0.0)
    
    # noise_std=0 makes capacity equal true_capacity
    pd.testing.assert_series_equal(df["capacity"], df["true_capacity"], check_names=False)

def test_invalid_parameters():
    # invalid n_cycles
    with pytest.raises(ValueError):
        simulate_battery_capacity("B01", n_cycles=1)
        
    # invalid initial_capacity
    with pytest.raises(ValueError):
        simulate_battery_capacity("B01", initial_capacity=0)
        
    # invalid noise_std
    with pytest.raises(ValueError):
        simulate_battery_capacity("B01", noise_std=-0.1)

def test_simulate_battery_population():
    n_batteries = 4
    n_cycles = 170
    df = simulate_battery_population(n_batteries=n_batteries, n_cycles=n_cycles)
    
    # produces requested number of unique batteries
    unique_batteries = df["battery_id"].unique()
    assert len(unique_batteries) == n_batteries
    
    # produces expected total row count
    assert len(df) == n_batteries * n_cycles
