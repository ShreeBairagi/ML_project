import pytest
import numpy as np
import pandas as pd

def test_weighted_cost_logic():
    cost_fp = 1.0
    cost_fn = 5.0
    
    fp = 3
    fn = 2
    
    weighted_cost = fp * cost_fp + fn * cost_fn
    assert weighted_cost == 13.0

def test_binary_target_creation():
    df = pd.DataFrame({"target_next_capacity": [1.3, 1.4, 1.5, 1.8]})
    df["next_cycle_below_eol"] = (df["target_next_capacity"] <= 1.4).astype(int)
    
    assert list(df["next_cycle_below_eol"]) == [1, 1, 0, 0]

def test_classification_thresholds():
    y_pred_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.9])
    
    y_pred_bin_50 = (y_pred_prob >= 0.5).astype(int)
    assert list(y_pred_bin_50) == [0, 0, 0, 1, 1]
    
    y_pred_bin_cost = (y_pred_prob >= (1.0/6.0)).astype(int)
    assert list(y_pred_bin_cost) == [0, 1, 1, 1, 1]

def test_regression_threshold():
    y_pred_reg = np.array([1.3, 1.4, 1.45, 1.8])
    y_pred_bin_ridge = (y_pred_reg <= 1.4).astype(int)
    assert list(y_pred_bin_ridge) == [1, 1, 0, 0]

def test_expected_metric_rows():
    batteries = ["B0005", "B0006", "B0007", "B0018"]
    conditions = ["direct_svc_threshold_0.5", "direct_svc_cost_threshold", "ridge_regress_then_threshold"]
    assert len(batteries) * len(conditions) == 12
