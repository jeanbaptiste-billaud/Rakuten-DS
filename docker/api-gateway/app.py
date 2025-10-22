import os
import logging
from typing import Dict
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration des URLs des services
PREPROCESSING_SERVICE_URL = os.getenv(
    "PREPROCESSING_SERVICE_URL", 
    "http://localhost:8001"
)
MODEL_SERVICE_URL = os.getenv(
    "MODEL_SERVICE_URL", 
    "http://localhost:8002"
)

app = FastAPI(
    title="Rakuten Text Classification API",
    description="API de classification de produits basée sur les descriptions textuelles",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Texte brut à classifier (description produit)")


class PredictionResponse(BaseModel):
    predicted_class: int = Field(..., description="Classe prédite (prdtypecode)")
    confidence: float = Field(..., description="Confiance de la prédiction (0-1)")
    probabilities: Dict[int, float] = Field(..., description="Probabilités par classe")


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "service": "Rakuten API Gateway",
        "status": "running",
        "endpoints": ["/predict", "/health"]
    }


@app.get("/health")
async def health_check():
    """Vérifie l'état de santé de tous les services."""
    health_status = {
        "api_gateway": "healthy",
        "preprocessing": "unknown",
        "model": "unknown"
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check preprocessing service
            try:
                response = await client.get(f"{PREPROCESSING_SERVICE_URL}/health")
                health_status["preprocessing"] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception as e:
                health_status["preprocessing"] = f"unhealthy: {str(e)}"
            
            # Check model service
            try:
                response = await client.get(f"{MODEL_SERVICE_URL}/health")
                health_status["model"] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception as e:
                health_status["model"] = f"unhealthy: {str(e)}"
    
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}")
    
    all_healthy = all(
        status == "healthy" 
        for status in health_status.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": health_status
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prédit la catégorie d'un produit à partir de sa description textuelle.
    
    Pipeline:
    1. Preprocessing: Nettoyage et lemmatisation du texte
    2. Model: Prédiction avec le modèle SVM
    """
    logger.info(f"Nouvelle requête de prédiction (texte: {len(request.text)} chars)")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Étape 1: Preprocessing
            logger.info("Envoi au service de preprocessing...")
            preprocess_response = await client.post(
                f"{PREPROCESSING_SERVICE_URL}/preprocess",
                json={"text": request.text}
            )
            
            if preprocess_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur preprocessing: {preprocess_response.text}"
                )
            
            preprocessed_data = preprocess_response.json()
            logger.info("✅ Preprocessing terminé")
            
            # Étape 2: Prediction
            logger.info("Envoi au service de prédiction...")
            model_response = await client.post(
                f"{MODEL_SERVICE_URL}/predict",
                json={"text_cleaned": preprocessed_data["text_cleaned"]}
            )
            
            if model_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erreur modèle: {model_response.text}"
                )
            
            prediction_data = model_response.json()
            logger.info(f"✅ Prédiction: classe {prediction_data['predicted_class']} (confiance: {prediction_data['confidence']:.2%})")
            
            return PredictionResponse(**prediction_data)
    
    except httpx.TimeoutException:
        logger.error("Timeout lors de la communication avec les services")
        raise HTTPException(
            status_code=504,
            detail="Timeout lors du traitement de la requête"
        )
    except httpx.RequestError as e:
        logger.error(f"Erreur de communication: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service indisponible: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)