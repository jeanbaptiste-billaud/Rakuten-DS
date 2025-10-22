import logging
import os
from typing import Dict
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chemin du modèle
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/SVM/model.pkl")

# Chargement du modèle au démarrage
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Modèle SVM chargé depuis {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
    model = None

app = FastAPI(
    title="Rakuten Model Service",
    description="Service d'inférence avec modèle SVM (TF-IDF + SVM)",
    version="1.0.0"
)


class PredictRequest(BaseModel):
    text_cleaned: str = Field(..., description="Texte prétraité (nettoyé et lemmatisé)")


class PredictResponse(BaseModel):
    predicted_class: int = Field(..., description="Classe prédite (prdtypecode)")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    probabilities: Dict[int, float] = Field(..., description="Probabilités par classe")


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "service": "Model Service",
        "status": "running",
        "model": "SVM (TF-IDF + RBF)",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Vérifie l'état de santé du service."""
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": "Modèle non chargé"
        }
    
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Prédit la classe d'un texte prétraité.
    
    Args:
        text_cleaned: Texte nettoyé et lemmatisé
    
    Returns:
        - predicted_class: Classe prédite (int)
        - confidence: Probabilité max (float)
        - probabilities: Dict {classe: probabilité}
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )
    
    if not request.text_cleaned or not request.text_cleaned.strip():
        raise HTTPException(
            status_code=400,
            detail="Texte vide après prétraitement"
        )
    
    logger.info(f"Prédiction pour un texte ({len(request.text_cleaned)} chars)")
    
    try:
        # Prédiction
        prediction = model.predict([request.text_cleaned])[0]
        probabilities = model.predict_proba([request.text_cleaned])[0]
        
        # Mapping des probabilités par classe
        classes = model.classes_
        prob_dict = {int(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        
        # Confiance = probabilité max
        confidence = float(probabilities.max())
        
        logger.info(f"✅ Classe prédite: {prediction} (confiance: {confidence:.2%})")
        
        return PredictResponse(
            predicted_class=int(prediction),
            confidence=confidence,
            probabilities=prob_dict
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)