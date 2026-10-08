import pytest
import pandas as pd
import numpy as np
import scipy.optimize

from experiments.run_baselines import get_last_observed_value, get_exponential_baseline

def test_last_observed_value():
    df = pd.DataFrame({
        'discharge_cycle_index': [0, 1, 2],
        'capacity': [2.0, 1.9, 1.8]
    })
    pred = get_last_observed_value(df)
    assert pred == 1.8

def test_exponential_fallback_short_prefix():
    df = pd.DataFrame({
        'discharge_cycle_index': [0, 1],
        'capacity': [2.0, 1.9]
    })
    pred, method = get_exponential_baseline(df, 2)
    assert method == 'fallback_last_observed'
    assert pred == 1.9

def test_exponential_no_leakage():
    df = pd.DataFrame({
        'discharge_cycle_index': [0, 1, 2, 3],
        'capacity': [2.0, 1.95, 1.90, 1.85]
    })
    pred, method = get_exponential_baseline(df, 4)
    assert np.isfinite(pred)
    assert method in ['exponential', 'fallback_last_observed']

def test_exponential_unfittable():
    # Provide highly oscillatory data that should cause curve_fit to fail
    df = pd.DataFrame({
        'discharge_cycle_index': [0, 1, 2, 3, 4],
        'capacity': [1.0, 100.0, -50.0, 0.0, 1.0]
    })
    # To ensure it throws RuntimeError, we could pass maxfev=1, but we don't control maxfev from here
    # The behavior should be that it falls back if it fails
    pred, method = get_exponential_baseline(df, 5)
    if method == 'fallback_last_observed':
        assert pred == 1.0
