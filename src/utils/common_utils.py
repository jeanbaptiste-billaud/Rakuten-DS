# common_utils.py
import os


def get_project_root():
    """Retourne le chemin absolu vers la racine du projet (Rakuten-DS)."""
    return os.path.dirname(os.path.abspath(os.path.join(__file__, "../..")))



#todo: ajouter une fonction de logging boto pour minio, voir artifact_test