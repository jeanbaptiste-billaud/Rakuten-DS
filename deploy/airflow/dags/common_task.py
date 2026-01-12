import os

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

# =============================================================================
# 🛠️ CONFIGURATION
# =============================================================================

MINIO_USER = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
WORKDIR = os.getenv("WORKDIR", "/workspace")
# DATA_VOLUME = "airflow_data"
# Configuration commune pour éviter de répéter le code dans chaque tâche

def docker_common_args():
    return {
        "api_version": "auto",
        "auto_remove": "success",
        "network_mode": "mlflow-network",
        "mounts": [
        # Mount(source=DATA_VOLUME, target="/workspace/data", type="volume"),
        ],
        "environment": {
            "WORKDIR": WORKDIR,
            # Config MLflow & MinIO
            "MINIO_HOST": "minio",
            "MINIO_PORT": "9000",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "MINIO_ACCESS_KEY": MINIO_USER,
            "MINIO_SECRET_KEY": MINIO_PASS,
            "AWS_ACCESS_KEY_ID": MINIO_USER,
            "AWS_SECRET_ACCESS_KEY": MINIO_PASS,
        }
    }


def preprocess_task():
    common_args = docker_common_args()
    return DockerOperator(
        task_id='preprocessing',
        image='jbbillaud/rakuten:spacy-v3.7.5',
        command="""sh -c '
            echo "⬇️ Downloading inputs..." &&
            python /src/utils/sync_bucket.py dataset --mode pull &&

            echo "⚙️ Processing Preprocessing..." &&
            python /src/preprocessing.py &&

            echo "⬆️ Uploading results..." &&
            python /src/utils/sync_bucket.py preprocessed --mode push
        '""",
        **common_args
    )


def start_pipeline_task():
    return BashOperator(
        task_id='start_pipeline',
        bash_command='echo "🚀 Démarrage du pipeline Rakuten"'
    )


def end_pipeline_task():
    return BashOperator(
        task_id='end_pipeline',
        bash_command='echo "✅ Pipeline terminé avec succès"'
    )
