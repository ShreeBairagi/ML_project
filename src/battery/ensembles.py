from sklearn.ensemble import VotingRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from src.battery.models import get_phase4_models

def get_phase6_base_models():
    """Return the 5 approved base models for Phase 6."""
    all_models = get_phase4_models()
    base_names = ["LinearRegression", "Ridge", "DecisionTreeRegressor", "KNeighborsRegressor", "SVR"]
    return {name: all_models[name] for name in base_names}

def get_phase6_sklearn_ensembles():
    """Return standard scikit-learn ensembles for Phase 6."""
    base_models = get_phase6_base_models()
    estimators = [(name, model) for name, model in base_models.items()]
    
    return {
        "VotingRegressor": VotingRegressor(estimators=estimators),
        "BaggingRegressor": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=5, random_state=42),
            n_estimators=25,
            random_state=42
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }

def get_phase6_weights():
    """Return the fixed weights for Weighted Averaging."""
    return {
        "LinearRegression": 0.25,
        "Ridge": 0.25,
        "DecisionTreeRegressor": 0.15,
        "KNeighborsRegressor": 0.15,
        "SVR": 0.20
    }
