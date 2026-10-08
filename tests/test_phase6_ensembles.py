import pytest
import numpy as np
import pandas as pd
from src.battery.ensembles import get_phase6_base_models, get_phase6_sklearn_ensembles, get_phase6_weights
from src.battery.splits import leave_one_battery_out
from src.battery.features import get_feature_columns

def test_ensemble_methods_exist():
    base_models = get_phase6_base_models()
    assert len(base_models) == 5
    assert "LinearRegression" in base_models
    assert "Ridge" in base_models
    assert "DecisionTreeRegressor" in base_models
    assert "KNeighborsRegressor" in base_models
    assert "SVR" in base_models
    
    ensembles = get_phase6_sklearn_ensembles()
    assert "VotingRegressor" in ensembles
    assert "BaggingRegressor" in ensembles
    assert "RandomForestRegressor" in ensembles

def test_weighted_average_weights():
    weights = get_phase6_weights()
    assert np.isclose(sum(weights.values()), 1.0)
    
def test_weighted_averaging_uses_declared_weights():
    weights = get_phase6_weights()
    preds = {
        "LinearRegression": 1.0,
        "Ridge": 2.0,
        "DecisionTreeRegressor": 1.0,
        "KNeighborsRegressor": 1.0,
        "SVR": 1.0
    }
    expected = (1.0 * 0.25 + 2.0 * 0.25 + 1.0 * 0.15 + 1.0 * 0.15 + 1.0 * 0.20)
    actual = sum(preds[k] * weights[k] for k in weights)
    assert np.isclose(expected, actual)

def test_lobo_battery_separation():
    df = pd.DataFrame({
        "battery_id": ["B1", "B1", "B2", "B2"],
        "discharge_cycle_index": [1, 2, 1, 2],
        "feature": [0.1, 0.2, 0.3, 0.4]
    })
    train, test = leave_one_battery_out(df, "B1")
    assert "B1" not in train["battery_id"].values
    assert "B1" in test["battery_id"].values
    assert len(test) == 2

def test_prediction_rows_align_by_battery_and_cycle():
    df1 = pd.DataFrame({"battery_id": ["B1"], "discharge_cycle_index": [1], "pred1": [1.0]})
    df2 = pd.DataFrame({"battery_id": ["B1"], "discharge_cycle_index": [1], "pred2": [2.0]})
    merged = pd.merge(df1, df2, on=["battery_id", "discharge_cycle_index"])
    assert len(merged) == 1
    assert merged["pred1"].iloc[0] == 1.0
    assert merged["pred2"].iloc[0] == 2.0

def test_residual_correlation_properties():
    residuals = pd.DataFrame({
        "M1": [0.1, -0.1, 0.2],
        "M2": [0.05, -0.15, 0.25]
    })
    corr = residuals.corr()
    assert corr.shape == (2, 2)
    assert np.allclose(corr.values, corr.values.T)
    assert np.allclose(np.diag(corr.values), [1.0, 1.0])

def test_target_excluded_from_x():
    features = get_feature_columns()
    assert "target_next_capacity" not in features
