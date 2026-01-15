# src/model_serving/build_step.py
import json
import os

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

    return model_uri


ROOT_PATH = os.getenv("WORKDIR", get_project_root())

if not os.getenv("MLFLOW_RUN_ID"):
    run_id_path = os.path.join(ROOT_PATH, "training_exports", "run_id.json")

    with open(run_id_path, "r") as f:
        meta = json.load(f)
    run_id = meta["run_id"]
    os.environ["MLFLOW_RUN_ID"] = run_id

model_uri = get_model_uri()
os.environ["MODEL_URI"] = model_uri
model_id = model_uri.removeprefix("models:/")

bento_model = bentoml.mlflow.import_model(
    name=f"rakuten_text_classifier:{run_id}",
    model_uri=model_uri,
    metadata={
        "mlflow_uri": model_uri,
        "mlflow.model_uri": model_uri,
        "model_id": model_uri.removeprefix("models:/"),
        "mlflow_run_id": run_id,
        "mlflow.run_id": run_id,
    },
)

print("Model registered as Bento:", bento_model)
