# src/model_serving/build_step.py

import os
import mlflow
import bentoml


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


model_uri = get_model_uri()
model_id = model_uri.removeprefix("models:/")
run_id = os.getenv("MLFLOW_RUN_ID")

bento_model = bentoml.mlflow.import_model(
    name=f"rakuten_text_classifier:{run_id}",
    model_uri=model_uri,
    metadata={"mlflow_uri": model_uri,
              "model_id": model_uri.removeprefix("models:/"),
              "mlflow_run_id": run_id,},
)


print("Model registered as Bento:", bento_model)




