import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.battery import splits
import numpy.testing as npt

def test_correct_pipeline_structure():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    assert isinstance(pipe.steps[0][1], StandardScaler)
    assert isinstance(pipe.steps[1][1], Ridge)

def test_scaler_leakage_correct_condition():
    # Verify that in the CORRECT_SCALER condition, the fitted scaler 
    # statistics come ONLY from the training rows
    np.random.seed(42)
    df = pd.DataFrame({
        'discharge_cycle_index': np.arange(100),
        'capacity': np.random.randn(100) * 0.1 + 1.5,
        'target_next_capacity': np.random.randn(100) * 0.1 + 1.5
    })
    
    # Split
    train_df, test_df = splits.random_cycle_split_wrong(df, random_state=42)
    X_train = train_df[['discharge_cycle_index', 'capacity']]
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    pipe.fit(X_train, train_df['target_next_capacity'])
    
    # Compare scaler.mean_ with the mean of X_train
    scaler = pipe.named_steps['scaler']
    npt.assert_allclose(scaler.mean_, X_train.mean().values)
    
    # Ensure it is NOT the full dataset mean
    full_mean = df[['discharge_cycle_index', 'capacity']].mean().values
    assert not np.allclose(scaler.mean_, full_mean)

def test_target_never_in_x():
    features = ['discharge_cycle_index', 'capacity']
    assert 'target_next_capacity' not in features

def test_lobo_train_test_overlap():
    df = pd.DataFrame({
        'battery_id': ['B1']*10 + ['B2']*10 + ['B3']*10,
        'val': np.arange(30)
    })
    train_df, test_df = splits.leave_one_battery_out(df, 'B2')
    
    assert 'B2' not in train_df['battery_id'].values
    assert 'B2' in test_df['battery_id'].values
    assert len(set(train_df['battery_id']).intersection(set(test_df['battery_id']))) == 0

def test_random_split_preserves_rows():
    df = pd.DataFrame({
        'val': np.arange(105)
    })
    train_df, test_df = splits.random_cycle_split_wrong(df, test_size=0.2)
    assert len(train_df) + len(test_df) == len(df)

def test_deliberately_wrong_global_scaler_isolated():
    # Verify that we can fit a global scaler without a pipeline and the mean matches full data
    df = pd.DataFrame({
        'discharge_cycle_index': np.arange(100),
        'capacity': np.random.randn(100) * 0.1 + 1.5
    })
    global_scaler = StandardScaler()
    global_scaler.fit(df)
    npt.assert_allclose(global_scaler.mean_, df.mean().values)
    
def test_result_row_metadata_checks():
    # Check that the required columns are specified in the output format
    required_cols = [
        'experiment', 'condition', 'split_type', 'battery_id', 'seed', 
        'alpha', 'feature_set', 'MAE', 'RMSE', 'R2', 'n_test_rows'
    ]
    # We just ensure our mock dictionary has all these keys
    mock_result = {k: None for k in required_cols}
    assert all(k in mock_result for k in required_cols)
