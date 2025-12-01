import os
import logging
from typing import Dict
import httpx

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
)

# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# CONFIG SERVICES
# ----------------------------------------------------
PREPROCESSING_SERVICE_URL = os.getenv(
    "PREPROCESSING_SERVICE_URL", 
    "http://localhost:8001"
)
MODEL_SERVICE_URL = os.getenv(
    "MODEL_SERVICE_URL", 
    "http://localhost:8002"
)

# ----------------------------------------------------
# FASTAPI
# ----------------------------------------------------
app = FastAPI(
    title="Rakuten Text Classification API",
    description="API de classification de produits basée sur les descriptions textuelles",
    version="1.0.0"
)

# ----------------------------------------------------
# METRICS PROMETHEUS
# ----------------------------------------------------
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

# ----------------------------------------------------
# MODELES Pydantic
# ----------------------------------------------------
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Texte brut à classifier (description produit)")


class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="Classe prédite (prdtypecode)")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    probabilities: Dict[int, float] = Field(..., description="Probabilités par classe")


# ----------------------------------------------------
# ROUTES API GATEWAY
# ----------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "Rakuten API Gateway",
        "status": "running",
        "endpoints": ["/predict", "/health", "/metrics"]
    }

@app.get("/metrics")
def metrics():
    """Expose les métriques Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    endpoint, method = "/health", "GET"
    API_REQUEST_COUNT.labels(endpoint, method).inc()

    with API_LATENCY.labels(endpoint, method).time():
        health_status = {
            "api_gateway": "healthy",
            "preprocessing": "unknown",
            "model": "unknown"
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:

                # Preprocessing
                try:
                    resp = await client.get(f"{PREPROCESSING_SERVICE_URL}/health")
                    health_status["preprocessing"] = "healthy" if resp.status_code == 200 else "unhealthy"
                except Exception as e:
                    health_status["preprocessing"] = f"unhealthy: {str(e)}"

                # Model
                try:
                    resp = await client.get(f"{MODEL_SERVICE_URL}/health")
                    health_status["model"] = "healthy" if resp.status_code == 200 else "unhealthy"
                except Exception as e:
                    health_status["model"] = f"unhealthy: {str(e)}"

        except Exception as e:
            logger.error(f"Erreur health check: {e}")

        all_ok = all(s == "healthy" for s in health_status.values())

        return {
            "status": "healthy" if all_ok else "degraded",
            "services": health_status
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    endpoint, method = "/predict", "POST"
    API_REQUEST_COUNT.labels(endpoint, method).inc()

    with API_LATENCY.labels(endpoint, method).time():
        logger.info(f"Nouvelle requête de prédiction ({len(request.text)} chars)")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:

                # Preprocessing
                prep = await client.post(
                    f"{PREPROCESSING_SERVICE_URL}/preprocess",
                    json={"text": request.text}
                )
                if prep.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"Erreur preprocessing: {prep.text}")

                data = prep.json()

                # Prediction
                model_resp = await client.post(
                    f"{MODEL_SERVICE_URL}/predict",
                    json={"text_cleaned": data["text_cleaned"]}
                )

                if model_resp.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"Erreur modèle: {model_resp.text}")

                result = model_resp.json()

                return PredictionResponse(**result)

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Timeout avec les services")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
