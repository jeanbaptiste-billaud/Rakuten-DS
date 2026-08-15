# src/model_serving/build_step.py

import json
import os
from pathlib import Path

import bentoml

import mlflow
from mlflow.tracking import MlflowClient
from src.utils.common_utils import get_project_root


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def resolve_run_id(workdir: str) -> str:
    """
    Résout le run_id MLflow :
    - priorité à MLFLOW_RUN_ID
    - fallback sur run_id.json dans le volume Airflow
    """
    resolved_run_id = os.getenv("MLFLOW_RUN_ID")
    if resolved_run_id:
        return resolved_run_id

    run_id_path = os.path.join(workdir, "training_exports", "run_id.json")
    with open(run_id_path, "r") as metadata_file:
        meta = json.load(metadata_file)

    resolved_run_id = meta["run_id"]
    os.environ["MLFLOW_RUN_ID"] = resolved_run_id
    return resolved_run_id


def get_model_uri_from_run(run_id: str) -> tuple[str, str]:
    """
    Récupère le model_uri depuis un run MLflow existant.
    """
    client = MlflowClient()
    run = client.get_run(run_id)

    if not run.outputs or not run.outputs.model_outputs:
        raise RuntimeError(f"Aucun model_output trouvé pour le run {run_id}")

    resolved_model_id = run.outputs.model_outputs[0].model_id
    resolved_model_uri = f"models:/{resolved_model_id}"

    return resolved_model_uri, resolved_model_id


# ---------------------------------------------------------------------
# Main build step
# ---------------------------------------------------------------------

# Résolution du workdir
workdir = os.getenv("WORKDIR", get_project_root())

# Tracking URI obligatoire ici (build-time)
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if not mlflow_uri:
    raise ValueError("MLFLOW_TRACKING_URI absente des variables d'environnement")

mlflow.set_tracking_uri(mlflow_uri)

# Résolution du run_id
resolved_run_id = resolve_run_id(workdir)

# Récupération du modèle MLflow
resolved_model_uri, resolved_model_id = get_model_uri_from_run(resolved_run_id)

# Récupération des métriques de performance du modèle
mlflow.artifacts.download_artifacts(run_id=resolved_run_id,
                                    artifact_path="eval/metrics_text.json",
                                    dst_path="/model_serving")

# Exports ENV (⚠️ NE PAS SUPPRIMER)
os.environ["MODEL_URI"] = resolved_model_uri

env_exports = {
    "MLFLOW_RUN_ID": resolved_run_id,
    "MODEL_URI": resolved_model_uri,
}

# ---------------------------------------------------------------------
# Chargement du modèle depuis MLflow
# ---------------------------------------------------------------------

loaded = mlflow.sklearn.load_model(resolved_model_uri)

# Cas où MLflow renvoie un wrapper
model = loaded.model if hasattr(loaded, "model") else loaded

# ---------------------------------------------------------------------
# Enregistrement BentoML (runtime sans MLflow)
# ---------------------------------------------------------------------

bento_model = bentoml.sklearn.save_model(
    name="rakuten_text_classifier",
    model=model,
    signatures={"predict": {"batchable": True}, },
    metadata={
        "mlflow.model_uri": resolved_model_uri,
        "mlflow.model_id": resolved_model_id,
        "mlflow.run_id": resolved_run_id,
    },
)

# ---------------------------------------------------------------------
# Traçabilité
# ---------------------------------------------------------------------

Path("bento_model_tag.txt").write_text(str(bento_model.tag))
Path("mlflow_run_id.txt").write_text(resolved_run_id)

with open(".env", "w") as f:
    for key, val in env_exports.items():
        f.write(f'export {key}="{val}"\n')

print("Model registered as BentoML:", bento_model)
print("BENTO_MODEL_TAG written to bento_model_tag.txt")
print(".env file written with exported variables")
