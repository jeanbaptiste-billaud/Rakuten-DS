import json
import hashlib
import os


def check_token(token: str, required_role: str = "user", config_path=None):
    """
    Vérifie le token par rapport au fichier JSON et au rôle requis.
    """
    # Si aucun chemin n'est fourni, utiliser le fichier auth.json du même dossier que ce script
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "auth.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Fichier d'authentification introuvable : {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Hash du token fourni par l’utilisateur
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # Recherche du rôle correspondant
    role = None
    for r, data in config["users"].items():
        if data["token_hash"] == token_hash:
            role = r
            break

    if not role:
        raise PermissionError("❌ Token invalide : accès refusé.")

    if required_role == "admin" and role != "admin":
        raise PermissionError("🚫 Accès refusé : rôle administrateur requis.")

    print(f"🔐 Authentification réussie, rôle : <{role}>.")
    return role


if __name__ == "__main__":
    try:
        token = input("Entrez votre token (123 pour admin, 456 pour user) : ")
        check_token(token)
        print()
    except Exception as e:
        print(e)
