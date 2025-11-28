# src/bentoml_module/build_step.py

import os
import mlflow
import bentoml


def get_model_uri():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    run_id = os.getenv("MLFLOW_RUN_ID")

    mlflow.set_tracking_uri(mlflow_uri)
    run = mlflow.get_run(run_id)

    model_id = run.outputs.model_outputs[0].model_id
    model_uri = f"runs:/{run_id}/model"

    return model_uri


model_uri = get_model_uri()
bento = bentoml.mlflow.import_model(
    name="rakuten_text_classifier",
    model_uri=model_uri,
    metadata={"mlflow_uri": model_uri, "model_version": 1})
# todo: modifier model version

model = bentoml.mlflow.load_model("rakuten_text_classifier:latest")


