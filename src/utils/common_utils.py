# common_utils.py
from pathlib import Path


def get_project_root():
    """Retourne le chemin absolu vers la racine du projet (Rakuten-DS)."""
    return Path(__file__).resolve().parents[2]



#todo: ajouter une fonction de logging boto pour minio, voir artifact_test