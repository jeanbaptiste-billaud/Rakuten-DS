from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
import os

# =============================================================================
# 🛠️ CONFIGURATION
# =============================================================================

# On récupère le chemin actuel (racine du projet dans le conteneur)
CURRENT_DIR = os.getcwd()

# Configuration commune pour éviter de répéter le code dans chaque tâche
DOCKER_COMMON_ARGS = {
    "image": "docker-compose-training:latest",  # Votre image locale
    "api_version": "auto",
    "auto_remove": True,          # Nettoie le conteneur après exécution
    "force_pull":  False,
    "docker_url": "unix://var/run/docker.sock",
    "network_mode": "mlflow-network", # Pour parler à MinIO et MLflow
    "working_dir": "/workspace",  # Dossier de travail DANS le conteneur éphémère
    "mounts": [
        # # On monte le code source pour qu'il soit à jour
        # Mount(source=f"{CURRENT_DIR}/src", target="/workspace/src", type="bind"),
        # On monte les données pour qu'elles persistent entre les étapes
        Mount(source=f"{CURRENT_DIR}/data", target="/workspace/data", type="bind"),
        # # On monte les configs
        # Mount(source=f"{CURRENT_DIR}/configs", target="/workspace/configs", type="bind"),
    ],
    "environment": {
        # Indispensable pour que python trouve le module 'src'
        "PYTHONPATH": "/workspace",
        # Config MLflow & MinIO
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",     # À ajuster si diff dans le .env
        "AWS_SECRET_ACCESS_KEY": "minioadmin", 
        "PYTHONUNBUFFERED": "1" # Pour voir les logs en direct dans Airflow
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
    dag_id='rakuten_ml_pipeline',
    default_args=default_args,
    schedule_interval=None, # Déclenchement manuel uniquement
    catchup=False,
    tags=['mlops', 'rakuten', 'docker']
) as dag:

    start = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "🚀 Démarrage du pipeline Rakuten"'
    )

    # --- Étape 1 : Enrichissement du dataset ---
    # Lit data/raw -> Écrit dans data/dataset (ou ailleurs selon votre logique)
    enrich_task = DockerOperator(
        task_id='enrich_dataset',
        command="python src/pipeline_steps_df/enrich_raw_dataset.py",
        **DOCKER_COMMON_ARGS
    )

    # --- Étape 2 : Preprocessing ---
    # Nettoyage, tokenization, préparation pour l'entraînement
    preprocess_task = DockerOperator(
        task_id='preprocessing',
        command="python src/pipeline_steps_df/preprocessing.py",
        **DOCKER_COMMON_ARGS
    )

    # --- Étape 3 : Entraînement et vérification du Drift ---
    # Entraîne le modèle et loggue les métriques dans MLflow
    train_task = DockerOperator(
        task_id='train_model',
        command="python src/pipeline_steps_df/train_text_model.py",
        **DOCKER_COMMON_ARGS
    )

    end = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "✅ Pipeline terminé avec succès"'
    )

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================
    
    start >> enrich_task >> preprocess_task >> train_task >> end