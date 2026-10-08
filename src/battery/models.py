from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_phase4_models():
    """
    Returns a dictionary of Phase 4 regression models.
    Scale-sensitive models include a StandardScaler step in a Pipeline.
    """
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.001, max_iter=10000))
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000))
        ]),
        "KNeighborsRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5))
        ]),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5, random_state=42),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=1.0))
        ])
    }
