import os
import logging
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

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
