import json
import os
from typing import Dict, Any

from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from src.utils.common_utils import get_project_root


# ------------------------------------------------------------------
# Helper MLflow
# ------------------------------------------------------------------

def get_run_metrics(client: MlflowClient, run_id: str) -> Dict[str, Any]:
    run = client.get_run(run_id)
    return {
        "run_id": run_id,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
        "artifact_uri": run.info.artifact_uri,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
    }


# ------------------------------------------------------------------
# Extraction minimale des métriques d’évaluation
# ------------------------------------------------------------------

def extract_eval_metrics(run_payload: Dict[str, Any]) -> Dict[str, float | None]:
    m = run_payload["metrics"]
    return {
        "accuracy": m.get("accuracy"),
        "weighted_f1": m.get("weighted_f1"),
        "macro_f1": m.get("macro_f1"),
        "mean_confidence": m.get("mean_confidence"),
    }


# ------------------------------------------------------------------
# Décision KISS : True = promote, False = keep baseline
# ------------------------------------------------------------------

def should_promote_model(
        baseline: Dict[str, float | None],
        candidate: Dict[str, float | None],
        max_drop_weighted_f1: float = 0.001,
        max_drop_accuracy: float = 0.002,
        review_as_promote: bool = False,
) -> bool:
    """
    Décide si le modèle candidate remplace la baseline.

    Règles :
    - weighted_f1 ne doit pas baisser au-delà du seuil
    - accuracy ne doit pas baisser au-delà du seuil
    - macro_f1 ne doit pas régresser
    - mean_confidence = signal faible (optionnel)
    """

    # Hard gates
    if (
            baseline["weighted_f1"] is not None
            and candidate["weighted_f1"] is not None
            and baseline["weighted_f1"] - candidate["weighted_f1"] > max_drop_weighted_f1
    ):
        return False

    if (
            baseline["accuracy"] is not None
            and candidate["accuracy"] is not None
            and baseline["accuracy"] - candidate["accuracy"] > max_drop_accuracy
    ):
        return False

    if (
            baseline["macro_f1"] is not None
            and candidate["macro_f1"] is not None
            and candidate["macro_f1"] < baseline["macro_f1"]
    ):
        return False

    # Soft signal : confidence
    if (
            baseline["mean_confidence"] is not None
            and candidate["mean_confidence"] is not None
            and baseline["mean_confidence"] - candidate["mean_confidence"] > 0.02
    ):
        return review_as_promote

    return True


# ------------------------------------------------------------------
# Main — DockerOperator / XCom friendly
# ------------------------------------------------------------------

if __name__ == "__main__":
    mlflow_client = MlflowClient()

    workdir = os.getenv("WORKDIR", "/app")
    run_id_path = os.path.join(workdir, "training_exports", "run_id.json")

    # Run ID du nouveau modèle (issu de l'entraînement)
    with open(run_id_path, "r") as f:
        new_run_id = json.load(f)["run_id"]

    # Run ID du modèle actuellement en service
    model_run_id = os.environ["MODEL_RUN_ID"]

    # Récupération MLflow
    new_model_metrics = get_run_metrics(mlflow_client, new_run_id)
    try:
        actual_model_metrics = get_run_metrics(mlflow_client, model_run_id)
        # Extraction métriques utiles
        new_eval = extract_eval_metrics(new_model_metrics)
        actual_eval = extract_eval_metrics(actual_model_metrics)

        # Décision
        promote = should_promote_model(
            baseline=actual_eval,
            candidate=new_eval,
            review_as_promote=False,
        )
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            promote = True
        else:
            raise

    # IMPORTANT : dernière ligne stdout pour XCom
    print(str(promote).lower())
