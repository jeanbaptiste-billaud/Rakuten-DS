import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG
from airflow.sdk import timezone

from common_task import start_pipeline_task, end_pipeline_task, preprocess_task, docker_common_args

# =============================================================================
# 🛠️ DÉFINITION DES TASK
# =============================================================================

def dataset_task():
    common_args = docker_common_args()
    return DockerOperator(
        task_id='create_dataset',
        image="jbbillaud/rakuten:spacy-v3.8.11",
        command="""sh -c '
            echo "⬇️ Downloading inputs..." &&
            python /src/utils/sync_bucket.py raw --mode pull &&

            echo "⚙️ Generate original dataset..." &&
            python /src/make_original_dataset.py &&

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
        dag_id='original_dataset_pipeline',
        default_args=default_args,
        catchup=False,
        tags=['mlops', 'rakuten', 'docker']
) as dag:
    start = start_pipeline_task()

    # --- Étape 1 : Création du dataset ---
    # Logique :
    # 1. Pull RAW
    # 2. Créer le dataset
    # 3. Push du dataset
    dataset_task = dataset_task()

    # --- Étape 2 : Preprocessing ---
    # Logique :
    # 1. Pull DATASET (celui qui vient d'être créé)
    # 2. Exécuter le preprocessing
    # 3. Push le résultat dans le bucket "preprocessed"
    preprocess_task = preprocess_task()

    end = end_pipeline_task()

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================

    start >> dataset_task >> preprocess_task >> end
