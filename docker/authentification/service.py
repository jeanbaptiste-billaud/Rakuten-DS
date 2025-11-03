from fastapi import FastAPI, Header, HTTPException
from auth_utils import check_token

app = FastAPI(title="Service d'authentification sécurisé")


@app.get("/")
def home():
    return {"message": "Bienvenue dans le service d'authentification 🎯"}


@app.get("/predict")
def predict(token: str = Header(...)):
    """
    Exemple d'endpoint protégé.
    Requiert un token valide dans le header HTTP.
    """
    try:
        role = check_token(token, required_role="user")
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

    return {
        "message": f"✅ Prédiction réussie pour le rôle <{role}>",
        "role": role
    }


@app.get("/admin-only")
def admin_action(token: str = Header(...)):
    """
    Exemple d'endpoint réservé aux admins.
    """
    try:
        role = check_token(token, required_role="admin")
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

    return {"message": f"🛡️ Action admin autorisée pour {role}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
