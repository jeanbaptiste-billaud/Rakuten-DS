import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

# Add docker/api-gateway to sys.path to allow importing app
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
api_gateway_path = os.path.join(project_root, "docker/api-gateway")
sys.path.append(api_gateway_path)

from app import app

client = TestClient(app)

@pytest.fixture
def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock_client:
        yield mock_client

def test_health_check_success(mock_httpx_client):
    """Test health check when model-serving is healthy."""
    # Setup mock response
    mock_instance = MagicMock()
    mock_instance.__aenter__.return_value = mock_instance
    mock_instance.__aexit__.return_value = None
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_instance.get = AsyncMock(return_value=mock_response)
    
    mock_httpx_client.return_value = mock_instance

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model": "healthy"}

def test_health_check_failure(mock_httpx_client):
    """Test health check when model-serving is unhealthy."""
    mock_instance = MagicMock()
    mock_instance.__aenter__.return_value = mock_instance
    mock_instance.__aexit__.return_value = None
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_instance.get = AsyncMock(return_value=mock_response)
    
    mock_httpx_client.return_value = mock_instance

    response = client.get("/health")
    assert response.status_code == 200 # It returns 200 but status degraded
    assert response.json() == {"status": "degraded", "model": "unhealthy"}

def test_predict_endpoint_success(mock_httpx_client):
    """Test predict endpoint with successful model response."""
    mock_instance = MagicMock()
    mock_instance.__aenter__.return_value = mock_instance
    mock_instance.__aexit__.return_value = None
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predicted_class": 1,
        "confidence": 0.95,
        "probabilities": {1: 0.95, 0: 0.05}
    }
    mock_instance.post = AsyncMock(return_value=mock_response)
    mock_httpx_client.return_value = mock_instance

    payload = {"text_cleaned": "test product"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["predicted_class"] == 1
    assert response.json()["confidence"] == 0.95

def test_trigger_pipeline_success(mock_httpx_client):
    """Test firing a pipeline trigger."""
    mock_instance = MagicMock()
    mock_instance.__aenter__.return_value = mock_instance
    mock_instance.__aexit__.return_value = None
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_instance.post = AsyncMock(return_value=mock_response)
    mock_httpx_client.return_value = mock_instance

    response = client.post("/pipelines/train")
    
    assert response.status_code == 200
    assert response.json()["status"] == "triggered"
    assert response.json()["dag_id"] == "rakuten_train_model"
