# src/model_serving/build_step.py
import json
import os
from pathlib import Path

import bentoml
import mlflow

from src.utils.common_utils import get_project_root


def get_model_uri():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    run_id = os.getenv("MLFLOW_RUN_ID")

    if not mlflow_uri or not run_id:
        raise ValueError("MLFLOW_TRACKING_URI ou MLFLOW_RUN_ID absent des variables d'environnement")

    mlflow.set_tracking_uri(mlflow_uri)
    run = mlflow.get_run(run_id)

    model_id = run.outputs.model_outputs[0].model_id
    model_uri = f"models:/{model_id}"

    return model_uri, model_id, run_id


workdir_path = os.getenv("WORKDIR", get_project_root())

env = dict()
# fallback run_id depuis fichier
if not os.getenv("MLFLOW_RUN_ID"):
    run_id_path = os.path.join(workdir_path, "training_exports", "run_id.json")
    with open(run_id_path, "r") as f:
        meta = json.load(f)
    os.environ["MLFLOW_RUN_ID"] = meta["run_id"]
    env["MLFLOW_RUN_ID"] = meta["run_id"]

model_uri, model_id, run_id = get_model_uri()
os.environ["MODEL_URI"] = model_uri
env["MODEL_URI"] = model_uri

# 1) Télécharger/charger l'objet sklearn depuis MLflow (build-only)
loaded = mlflow.sklearn.load_model(model_uri)
# si MLflow renvoie ton wrapper
pipeline = loaded.model if hasattr(loaded, "model") else loaded

# 2) Enregistrer en format BentoML sklearn (runtime sans mlflow)
bento_model = bentoml.sklearn.save_model(
    name="rakuten_text_classifier",
    model=pipeline,
    signatures={"predict": {"batchable": True}},
    metadata={
        "mlflow.model_uri": model_uri,
        "model_id": model_id,
        "mlflow_run_id": run_id,
        "mlflow.run_id": run_id,
    },
)

# Traçabilité facile à relire dans le service
Path("bento_model_tag.txt").write_text(str(bento_model.tag))
Path("mlflow_run_id.txt").write_text(run_id)
with open(".env", "w") as f:
    for key, val in env.items():
        f.write(f'export {key}="{val}"\n')

print("Model registered as Bento:", bento_model)
print("BENTO_MODEL_TAG written to bento_model_tag.txt")
