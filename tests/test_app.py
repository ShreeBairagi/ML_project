import pytest
from unittest.mock import patch
from app.app import safe_load_csv, safe_load_image, RESULTS_DIR

def test_paths_defined():
    assert RESULTS_DIR.name == "results"

@patch("app.app.st.warning")
def test_safe_load_csv_missing(mock_warning):
    # Pass a non-existent file
    df = safe_load_csv("non_existent_file_123.csv")
    assert df is None
    mock_warning.assert_called_once()
    assert "File not found" in mock_warning.call_args[0][0]

@patch("app.app.st.warning")
def test_safe_load_image_missing(mock_warning):
    img = safe_load_image("non_existent_image_123.png")
    assert img is None
    mock_warning.assert_called_once()
    assert "Image not found" in mock_warning.call_args[0][0]
