import os
import math
import logging
from typing import Dict
import joblib
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# -------------------------------------
# Logging
# -------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------
# Load Model
# -------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/SVM/model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Modèle SVM chargé depuis {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Erreur de chargement modèle: {e}")
    model = None

# -------------------------------------
# FastAPI
# -------------------------------------
app = FastAPI(title="Rakuten Model Service")

class PredictRequest(BaseModel):
    text_cleaned: str

class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: Dict[int, float]


# -------------------------------------
# PROMETHEUS METRICS
# -------------------------------------

# Total predictions
model_requests_total = Counter(
    "model_requests_total",
    "Nombre total de prédictions effectuées",
    ["predicted_class"]
)

# Confidence histogram
model_confidence_hist = Histogram(
    "model_prediction_confidence",
    "Histogramme des niveaux de confiance",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Prediction entropy (drift indicator)
model_entropy_hist = Histogram(
    "model_prediction_entropy",
    "Entropie des distributions de probabilité des prédictions (drift monitoring)",
    buckets=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]
)

# Rolling entropy (gauge)
model_entropy_gauge = Gauge(
    "model_entropy_last_value",
    "Valeur courante de l'entropie (drift potentiel)"
)


# -------------------------------------
# Helper: compute entropy
# -------------------------------------
def entropy(probabilities):
    return -sum(p * math.log(p + 1e-12) for p in probabilities)


# -------------------------------------
# Endpoints
# -------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):

    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    probs = model.predict_proba([request.text_cleaned])[0]
    prediction = model.classes_[probs.argmax()]
    confidence = float(probs.max())

    # Compute entropy
    ent = entropy(probs)

    # Update metrics
    model_requests_total.labels(predicted_class=int(prediction)).inc()
    model_confidence_hist.observe(confidence)
    model_entropy_hist.observe(ent)
    model_entropy_gauge.set(ent)

    return PredictResponse(
        predicted_class=int(prediction),
        confidence=confidence,
        probabilities={int(c): float(p) for c, p in zip(model.classes_, probs)}
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
