import sys
import os
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add docker/drift-detector to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
drift_path = os.path.join(project_root, "docker/drift-detector")
sys.path.append(drift_path)

# Mock Evidently and Path.mkdir before importing service to avoid heavy loading and permission errors
with patch.dict(sys.modules, {"evidently": MagicMock(), "evidently.presets": MagicMock()}):
    with patch("pathlib.Path.mkdir"): # Prevent /app creation attempt
        from service import app

# We need to re-patch commonly used objects inside service because of the import
from service import EvaluationData

client = TestClient(app)

@pytest.fixture
def mock_filesystem(tmp_path):
    """Mock STORAGE_DIR to use tmp_path."""
    with patch("service.STORAGE_DIR", tmp_path), \
         patch("service.REFERENCE_PATH", tmp_path / "reference_data.csv"), \
         patch("service.HISTORY_PATH", tmp_path / "metrics_history.json"):
        yield tmp_path

@patch("service.generate_drift_report")
def test_log_evaluation_first_run(mock_gen_report, mock_filesystem):
    """Test that first run creates baseline and does not detect drift."""
    
    payload = {
        "run_id": "run_1",
        "timestamp": "2023-01-01T12:00:00",
        "y_true": [1, 0],
        "y_pred": [1, 0],
        "metrics": {"accuracy": 1.0}
    }
    
    response = client.post("/log-evaluation", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] is False
    assert data["report_path"] == "N/A (baseline)"
    
    # Check that reference file was created
    assert (mock_filesystem / "reference_data.csv").exists()

@patch("service.generate_drift_report")
def test_log_evaluation_subsequent_run_no_drift(mock_gen_report, mock_filesystem):
    """Test subsequent run without drift."""
    # 1. Create baseline
    ref_path = mock_filesystem / "reference_data.csv"
    import pandas as pd
    pd.DataFrame({"target": [1], "prediction": [1], "run_id": "run_0"}).to_csv(ref_path, index=False)
    
    mock_gen_report.return_value = "/tmp/report.html"

    payload = {
        "run_id": "run_2",
        "timestamp": "2023-01-02T12:00:00",
        "y_true": [1],
        "y_pred": [1],
        "metrics": {"accuracy": 1.0}
    }
    
    response = client.post("/log-evaluation", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] is False
    assert data["accuracy_drop"] == 0.0

@patch("service.generate_drift_report")
def test_log_evaluation_with_drift(mock_gen_report, mock_filesystem):
    """Test subsequent run WITH drift."""
    # 1. Baseline has 100% accuracy
    ref_path = mock_filesystem / "reference_data.csv"
    import pandas as pd
    pd.DataFrame({"target": [1, 0], "prediction": [1, 0], "run_id": "run_0"}).to_csv(ref_path, index=False)

    mock_gen_report.return_value = "/tmp/report.html"

    # 2. Current run has 0% accuracy
    payload = {
        "run_id": "run_drift",
        "timestamp": "2023-01-03T12:00:00",
        "y_true": [1, 0],
        "y_pred": [0, 1], # All wrong
        "metrics": {"accuracy": 0.0}
    }
    
    # Threshold is 0.05 (5%), drop is 1.0 (100%) -> Drift
    response = client.post("/log-evaluation?drift_threshold=0.05", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] is True
    assert data["accuracy_drop"] == 1.0
