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
MINIO_USER = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
WORKDIR = os.getenv("WORKDIR", "/workspace")
DATA_VOLUME = "airflow_data"
# Configuration commune pour éviter de répéter le code dans chaque tâche
DOCKER_COMMON_ARGS = {
    "image": "spacy:3.7.5",  # Votre image locale
    "api_version": "auto",
    "auto_remove": True,          # Nettoie le conteneur après exécution
    "force_pull":  False,
    "docker_url": "unix://var/run/docker.sock",
    "network_mode": "mlflow-network", # Pour parler à MinIO et MLflow
    "working_dir": "/workspace",  # Dossier de travail DANS le conteneur éphémère
    "mounts": [
        Mount(source=DATA_VOLUME, target="/workspace/data", type="volume"),
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
        "AWS_ACCESS_KEY_ID": "minio",     # À ajuster si diff dans le .env
        "AWS_SECRET_ACCESS_KEY": "minio123", 
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

    init_volume = DockerOperator(
        task_id='init_volume_permissions',
        image="spacy:3.7.5",
        
        user='root', 
        
        # On donne le dossier à l'utilisateur 1000 (trainusr)
        command="chown -R 1000:1000 /workspace/data",
        
        # On monte le volume pour agir dessus
        mounts=DOCKER_COMMON_ARGS["mounts"],
        
        # Pas besoin du réseau ou des variables d'env complexes pour un chown
        api_version='auto',
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
    )
    # --- Étape 1 : Récupérer les données (remplace DVC ou mc mirror)
    # On lance le script sync_bucket.py en mode PULL
    # Cela va télécharger le bucket 'raw' vers /workspace/data/raw
    fetch_data_task = DockerOperator(
        task_id='fetch_raw_data',
        command="sh -c 'python /src/utils/sync_bucket.py raw --mode pull && python /src/utils/sync_bucket.py dataset --mode pull'",
        **DOCKER_COMMON_ARGS
    ) 

    # --- Étape 2 : Enrichissement du dataset ---
    # Lit data/raw -> Écrit dans data/dataset (ou ailleurs selon votre logique)
    enrich_task = DockerOperator(
        task_id='enrich_dataset',
        command="python /src/enrich_raw_dataset.py",
        **DOCKER_COMMON_ARGS
    )

    # --- Étape 3 : Preprocessing ---
    # Nettoyage, tokenization, préparation pour l'entraînement
    preprocess_task = DockerOperator(
        task_id='preprocessing',
        command="python /src/preprocessing.py",
        **DOCKER_COMMON_ARGS
    )

    # # --- Étape 4 : Entraînement et vérification du Drift ---
    # # Entraîne le modèle et loggue les métriques dans MLflow
    # train_task = DockerOperator(
    #     task_id='train_model',
    #     command="python src/pipeline_steps_df/training/train_text_model.py",
    #     **DOCKER_COMMON_ARGS
    # )

    # --- Étape 5 : Upload des résultats vers le bucket ---
    # On envoie /workspace/data/dataset vers le bucket 'dataset'
    upload_results_task = DockerOperator(
        task_id='upload_results',
        command="python /src/utils/sync_bucket.py dataset --mode push",
        **DOCKER_COMMON_ARGS
    )

    end = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "✅ Pipeline terminé avec succès"'
    )

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================
    
    start >> init_volume >> fetch_data_task >> enrich_task >> preprocess_task >> upload_results_task >> end