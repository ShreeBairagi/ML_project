import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.battery.models import get_phase4_models
from src.battery.features import get_feature_columns
from src.battery.splits import leave_one_battery_out
import pandas as pd

def test_phase4_models_presence():
    models = get_phase4_models()
    expected_models = {
        "LinearRegression", "Ridge", "Lasso", "ElasticNet",
        "KNeighborsRegressor", "DecisionTreeRegressor", "SVR"
    }
    assert set(models.keys()) == expected_models

def test_phase4_scaling():
    models = get_phase4_models()
    
    scale_sensitive = ["Ridge", "Lasso", "ElasticNet", "KNeighborsRegressor", "SVR"]
    for name in scale_sensitive:
        model = models[name]
        assert isinstance(model, Pipeline), f"{name} must be a Pipeline"
        assert isinstance(model.steps[0][1], StandardScaler), f"{name} must have StandardScaler as the first step"
        
    dt = models["DecisionTreeRegressor"]
    assert not isinstance(dt, Pipeline) or not isinstance(dt.steps[0][1], StandardScaler), "DecisionTreeRegressor must be unscaled"

def test_phase4_features_and_target():
    features = get_feature_columns()
    assert "target_next_capacity" not in features, "target_next_capacity must not be in features list"
    assert "discharge_cycle_index" in features
    assert "capacity" in features

def test_lobo_split_overlap():
    df = pd.DataFrame({
        "battery_id": ["B0005", "B0005", "B0006", "B0006", "B0007"],
        "capacity": [2.0, 1.9, 2.0, 1.9, 1.8]
    })
    
    train_df, test_df = leave_one_battery_out(df, "B0006")
    
    train_bats = set(train_df["battery_id"].unique())
    test_bats = set(test_df["battery_id"].unique())
    
    assert "B0006" not in train_bats
    assert "B0006" in test_bats
    assert train_bats.isdisjoint(test_bats), "Train and test battery IDs must not overlap"
