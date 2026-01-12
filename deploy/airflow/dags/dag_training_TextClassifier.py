from airflow.sdk import DAG
from airflow.sdk import timezone
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.bash import BashOperator
from common_task import start_pipeline_task, end_pipeline_task
from docker.types import Mount
import os

# =============================================================================
# 🛠️ CONFIGURATION
# =============================================================================

MINIO_USER = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
WORKDIR = os.getenv("WORKDIR", "/workspace")
ML_FLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

# Configuration commune pour éviter de répéter le code dans chaque tâche
DOCKER_COMMON_ARGS = {
    "api_version": "auto",
    "auto_remove": "force",  # Nettoie le conteneur après exécution
    "network_mode": "mlflow-network",  # Pour parler à MinIO et MLflow
    # "working_dir": "/workspace",  # Dossier de travail DANS le conteneur éphémère
    "mounts": [
        # Mount(source=DATA_VOLUME, target="/workspace/data", type="volume"),
    ],
    "environment": {
        "WORKDIR": WORKDIR,
        # Config MLflow & MinIO
        "MINIO_HOST": "minio",
        "MINIO_PORT": "9000",
        "MLFLOW_TRACKING_URI": ML_FLOW_TRACKING_URI,
        "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
        "MINIO_ACCESS_KEY": MINIO_USER,
        "MINIO_SECRET_KEY": MINIO_PASS,
        "AWS_ACCESS_KEY_ID": MINIO_USER,
        "AWS_SECRET_ACCESS_KEY": MINIO_PASS,
    }
}


# =============================================================================
#  DÉFINITION DES TASK
# =============================================================================

def training_task():
    return DockerOperator(
        task_id='train_model',
        image="jbbillaud/rakuten:sklearn-v1.7.2",
        command="""sh -c '
            echo "⬇️ Downloading preprocessed data..." &&
            python /src/utils/sync_bucket.py preprocessed --mode pull &&
            
            echo "🧠 Training model..." &&
            python /src/train_text_model_with_drift.py
        '""",
        **DOCKER_COMMON_ARGS
    )


# =============================================================================
# 🚀 DÉFINITION DU DAG
# =============================================================================

default_args = {
    'owner': 'rakuten-team',
    'start_date': timezone.datetime(2025, 1, 1),
    'retries': 0,  # Pas de retry pour le debug, on veut voir l'erreur tout de suite
}

with DAG(
        dag_id='training_pipeline',
        default_args=default_args,
        catchup=False,
        tags=['mlops', 'rakuten', 'docker', 'training']
) as dag:
    start = start_pipeline_task()

    # --- Tâche Unique : Pull Data -> Train -> Log to MLflow ---
    # On chaîne les commandes pour tout faire dans le même conteneur.
    # Note : On suppose que train_text_model.py gère l'upload du modèle via mlflow.log_model()
    # Si vous avez besoin de sauver un fichier spécifique hors MLflow, ajoutez un push à la fin.
    training_task = training_task()

    end = end_pipeline_task()

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================

    start >> training_task >> end
