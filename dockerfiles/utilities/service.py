import logging
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chargement du modèle spaCy
try:
    nlp = spacy.load("fr_core_news_sm")
    logger.info("✅ Modèle spaCy chargé avec succès")
except OSError:
    logger.error("❌ Modèle spaCy non trouvé")
    raise

app = FastAPI(
    title="Rakuten Preprocessing Service",
    description="Service de prétraitement de texte (nettoyage, lemmatisation)",
    version="1.0.0"
)


class PreprocessRequest(BaseModel):
    text: str = Field(..., description="Texte brut à prétraiter")


class PreprocessResponse(BaseModel):
    text_cleaned: str = Field(..., description="Texte nettoyé et lemmatisé")
    token_count: int = Field(..., description="Nombre de tokens après nettoyage")


def preprocess_text(text: str) -> str:
    """
    Nettoie et lemmatise une chaîne de texte.
    
    Étapes:
    - Suppression des stopwords français
    - Lemmatisation
    - Conversion en minuscules
    - Conservation uniquement des tokens alphabétiques
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    
    return " ".join(tokens)


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "service": "Preprocessing Service",
        "status": "running",
        "model": "spaCy fr_core_news_sm"
    }


@app.get("/health")
async def health_check():
    """Vérifie l'état de santé du service."""
    return {
        "status": "healthy",
        "model_loaded": nlp is not None
    }


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(request: PreprocessRequest):
    """
    Prétraite un texte brut.
    
    Returns:
        - text_cleaned: Texte nettoyé et lemmatisé
        - token_count: Nombre de tokens après traitement
    """
    logger.info(f"Prétraitement d'un texte ({len(request.text)} chars)")
    
    try:
        cleaned_text = preprocess_text(request.text)
        token_count = len(cleaned_text.split())
        
        logger.info(f"✅ Texte prétraité: {token_count} tokens")
        
        return PreprocessResponse(
            text_cleaned=cleaned_text,
            token_count=token_count
        )
    
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prétraitement: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)