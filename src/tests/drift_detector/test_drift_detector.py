import os
import sys
import importlib.util
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
service_path = os.path.join(project_root, "docker/drift-detector/service.py")

# Le conteneur utilise /app par défaut, mais ce chemin n'est pas inscriptible
# quand le test est lancé directement sur la machine de développement.
os.environ.setdefault("WORKDIR", tempfile.mkdtemp(prefix="rakuten-drift-test-"))

service_spec = importlib.util.spec_from_file_location(
    "drift_detector_service",
    service_path,
)
if service_spec is None or service_spec.loader is None:
    raise ImportError(f"Impossible de charger le service depuis {service_path}")

service = importlib.util.module_from_spec(service_spec)
sys.modules[service_spec.name] = service
service_spec.loader.exec_module(service)

app = service.app
EvaluationData = service.EvaluationData

client = TestClient(app)

@pytest.fixture
def mock_filesystem(tmp_path):
    """Mock STORAGE_DIR to use tmp_path."""
    with patch.dict(service.__dict__, {
        "STORAGE_DIR": tmp_path,
        "REFERENCE_PATH": tmp_path / "reference_data.csv",
        "HISTORY_PATH": tmp_path / "metrics_history.json",
    }):
        yield tmp_path

@pytest.fixture
def mock_generate_drift_report():
    mock_gen_report = MagicMock(return_value="/tmp/report.html")
    with patch.dict(service.__dict__, {
        "generate_drift_report": mock_gen_report,
    }):
        yield mock_gen_report


def test_log_evaluation_first_run(mock_filesystem):
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

def test_log_evaluation_subsequent_run_no_drift(
    mock_filesystem,
    mock_generate_drift_report,
):
    """Test subsequent run without drift."""
    # 1. Create baseline
    ref_path = mock_filesystem / "reference_data.csv"
    import pandas as pd
    pd.DataFrame({"target": [1], "prediction": [1], "run_id": "run_0"}).to_csv(ref_path, index=False)
    
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

def test_log_evaluation_with_drift(mock_filesystem, mock_generate_drift_report):
    """Test subsequent run WITH drift."""
    # 1. Baseline has 100% accuracy
    ref_path = mock_filesystem / "reference_data.csv"
    import pandas as pd
    pd.DataFrame({"target": [1, 0], "prediction": [1, 0], "run_id": "run_0"}).to_csv(ref_path, index=False)

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
