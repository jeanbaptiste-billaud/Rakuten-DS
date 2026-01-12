import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG
from airflow.sdk import timezone

from common_task import start_pipeline_task, end_pipeline_task, preprocess_task, docker_common_args

# =============================================================================
# 🛠️ DÉFINITION DES TASK
# =============================================================================

def enrich_task():
    common_args = docker_common_args()
    return DockerOperator(
        task_id='enrich_dataset',
        image="jbbillaud/rakuten:spacy-v3.7.5",
        command="""sh -c '
            echo "⬇️ Downloading inputs..." &&
            python /src/utils/sync_bucket.py raw --mode pull &&
            python /src/utils/sync_bucket.py dataset --mode pull &&
            
            echo "⚙️ Processing Enrich..." &&
            python /src/enrich_raw_dataset.py &&
            
            echo "⬆️ Uploading results..." &&
            python /src/utils/sync_bucket.py dataset --mode push
        '""",
        **common_args
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
        dag_id='enrich_dataset_pipeline',
        default_args=default_args,
        catchup=False,
        tags=['mlops', 'rakuten', 'docker']
) as dag:
    start = start_pipeline_task()

    # --- Étape 1 : Enrichissement ---
    # Logique :
    # 1. Pull RAW
    # 2. Pull DATASET (existant)
    # 3. Exécuter le script d'enrichissement
    # 4. Push le résultat dans DATASET
    enrich_task = enrich_task()

    # --- Étape 2 : Preprocessing ---
    # Logique :
    # 1. Pull DATASET (celui qui vient d'être mis à jour par l'étape précédente)
    # 2. Exécuter le preprocessing
    # 3. Push le résultat dans un NOUVEAU bucket "preprocessed"
    preprocess_task = preprocess_task()

    end = end_pipeline_task()

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================

    start >> enrich_task >> preprocess_task >> end
