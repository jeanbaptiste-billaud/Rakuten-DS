from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
import os

# =============================================================================
# 🛠️ CONFIGURATION
# =============================================================================

MINIO_USER = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
WORKDIR = os.getenv("WORKDIR", "/workspace")
# DATA_VOLUME = "airflow_data"
# Configuration commune pour éviter de répéter le code dans chaque tâche
DOCKER_COMMON_ARGS = {
    "image": "spacy:3.7.5",
    "api_version": "auto",
    "auto_remove": True,
    "force_pull":  False,
    "docker_url": "unix://var/run/docker.sock",
    "network_mode": "mlflow-network",
    "working_dir": "/workspace",  # Dossier de travail DANS le conteneur éphémère
    "mounts": [
        # Mount(source=DATA_VOLUME, target="/workspace/data", type="volume"),
    ],
    "user":"trainusr",
    "environment": {
        # Indispensable pour que python trouve le module 'src'
        "WORKDIR": WORKDIR,
        "PYTHONPATH": "/",
        # Config MLflow & MinIO
        "MINIO_HOST": "minio",
        "MINIO_PORT": "9000",
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
        "MINIO_ACCESS_KEY": MINIO_USER,
        "MINIO_SECRET_KEY": MINIO_PASS,
        "AWS_ACCESS_KEY_ID": MINIO_USER,     
        "AWS_SECRET_ACCESS_KEY": MINIO_PASS,
        "PYTHONUNBUFFERED": "1"
    }
}

# =============================================================================
# 🚀 DÉFINITION DU DAG
# =============================================================================

default_args = {
    'owner': 'rakuten-team',
    'start_date': days_ago(1),
    'retries': 0, # Pas de retry pour le debug, on veut voir l'erreur tout de suite
}

with DAG(
    dag_id='preprocessing_pipeline',
    default_args=default_args,
    schedule_interval=None, # Déclenchement manuel uniquement
    catchup=False,
    tags=['mlops', 'rakuten', 'docker']
) as dag:

    start = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "🚀 Démarrage du pipeline Rakuten"'
    )

    # --- Étape 1 : Enrichissement ---
    # Logique :
    # 1. Pull RAW
    # 2. Pull DATASET (existant)
    # 3. Exécuter le script d'enrichissement
    # 4. Push le résultat dans DATASET
    enrich_task = DockerOperator(
        task_id='enrich_dataset',
        command="""sh -c '
            echo "⬇️ Downloading inputs..." &&
            python /src/utils/sync_bucket.py raw --mode pull &&
            python /src/utils/sync_bucket.py dataset --mode pull &&
            
            echo "⚙️ Processing Enrich..." &&
            python /src/enrich_raw_dataset.py &&
            
            echo "⬆️ Uploading results..." &&
            python /src/utils/sync_bucket.py dataset --mode push
        '""",
        **DOCKER_COMMON_ARGS
    )

    # --- Étape 2 : Preprocessing ---
    # Logique :
    # 1. Pull DATASET (celui qui vient d'être mis à jour par l'étape précédente)
    # 2. Exécuter le preprocessing
    # 3. Push le résultat dans un NOUVEAU bucket "preprocessed"
    preprocess_task = DockerOperator(
        task_id='preprocessing',
        command="""sh -c '
            echo "⬇️ Downloading inputs..." &&
            python /src/utils/sync_bucket.py dataset --mode pull &&
            
            echo "⚙️ Processing Preprocessing..." &&
            python /src/preprocessing.py &&
            
            echo "⬆️ Uploading results..." &&
            python /src/utils/sync_bucket.py preprocessed --mode push
        '""",
        **DOCKER_COMMON_ARGS
    )

    end = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "✅ Pipeline terminé avec succès"'
    )

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================
    
    start >> enrich_task >> preprocess_task >> end