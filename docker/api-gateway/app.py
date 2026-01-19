import os
import logging
import httpx

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-apiserver:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ IMPORTANT: base URL, SANS /metrics
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-serving:8002")

app = FastAPI(title="Rakuten API Gateway")

API_REQUEST_COUNT = Counter(
    "api_gateway_requests_total",
    "Nombre total de requêtes reçues par l'API Gateway",
    ["endpoint", "method"]
)
API_LATENCY = Histogram(
    "api_gateway_request_latency_seconds",
    "Latence des requêtes de l'API Gateway",
    ["endpoint", "method"]
)


class PredictRequest(BaseModel):
    text_cleaned: str = Field(..., description="Texte déjà nettoyé")


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: Dict[int, float]


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    endpoint, method = "/predict", "POST"
    API_REQUEST_COUNT.labels(endpoint, method).inc()

    with API_LATENCY.labels(endpoint, method).time():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{MODEL_SERVICE_URL}/predict",
                    json={"text_cleaned": request.text_cleaned},
                )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"model-serving error: {resp.text}")
            return resp.json()

        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"model-serving unreachable: {e}")


@app.get("/health")
async def health_check():
    # Le model-serving de ton repo expose /metrics (pas /health) -> on ping /metrics
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{MODEL_SERVICE_URL}/metrics")
        model_ok = (r.status_code == 200)
    except Exception as e:
        return {"status": "degraded", "model": f"unhealthy: {e}"}

    return {"status": "healthy" if model_ok else "degraded", "model": "healthy" if model_ok else "unhealthy"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/pipelines/{pipeline_name}")
async def trigger_pipeline(pipeline_name: str):
    endpoint, method = f"/pipelines/{pipeline_name}", "POST"
    API_REQUEST_COUNT.labels(endpoint, method).inc()

    PIPELINE_TO_DAG_ID = {
        "backup": "rakuten_backup_pipeline",
        "enrich_dataset": "rakuten_enrich_dataset",
        "original_dataset": "rakuten_create_dataset",
        "train": "rakuten_train_model",
    }

    dag_id = PIPELINE_TO_DAG_ID.get(pipeline_name)
    if not dag_id:
        raise HTTPException(status_code=404, detail="Pipeline inconnu")

    dag_run_id = f"api__{pipeline_name}__{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"

    payload = {"dag_run_id": dag_run_id}

    with API_LATENCY.labels(endpoint, method).time():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{AIRFLOW_API_URL}/api/v2/dags/{dag_id}/dagRuns",
                    json=payload,
                    auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
                )

            if resp.status_code not in (200, 201):
                raise HTTPException(status_code=502, detail=f"Airflow error: {resp.text}")

            return {
                "status": "triggered",
                "pipeline": pipeline_name,
                "dag_id": dag_id,
                "dag_run_id": dag_run_id
            }

        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Airflow unreachable: {e}")


@app.get("/debug/routes")
def debug_routes():
    return [{"path": r.path, "name": r.name} for r in app.routes]
