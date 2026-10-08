"""Smoke tests for environment and package sanity."""

import sys
import pytest

def test_python_version():
    """Verify runtime is Python 3.11."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11

def test_core_dependencies_import():
    """Verify all syllabus dependencies are installed and importable."""
    import numpy as np
    import scipy
    import pandas as pd
    import matplotlib
    import seaborn as sns
    import sklearn
    import xgboost as xgb
    import streamlit as st

    assert np.__version__ is not None
    assert scipy.__version__ is not None
    assert pd.__version__ is not None
    assert matplotlib.__version__ is not None
    assert sns.__version__ is not None
    assert sklearn.__version__ is not None
    assert xgb.__version__ is not None
    assert st.__version__ is not None
    assert pytest.__version__ is not None
