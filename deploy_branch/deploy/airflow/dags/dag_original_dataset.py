from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import timezone, dag

from common_task import (
    docker_common_args,
    end_pipeline_task,
    infisical_run_command,
    preprocess_task,
    start_pipeline_task,
)


# =============================================================================
# 🛠️ DÉFINITION DES TASK
# =============================================================================

def dataset_task():
    identity = "data-ingestion-id"
    common_args = docker_common_args(identity)
    script = """
        echo "⬇️ Downloading inputs..." &&
        python /src/utils/sync_bucket.py raw --mode pull &&

        echo "⚙️ Generate original dataset..." &&
        python /src/make_original_dataset.py &&

        echo "⬆️ Uploading results..." &&
        python /src/utils/sync_bucket.py dataset --mode push
    """
    return DockerOperator(
        task_id='create_dataset',
        image="jbbillaud/rakuten:spacy-v3.8.11",
        command=infisical_run_command(script, identity),
        **common_args
    )


# =============================================================================
# 🚀 DÉFINITION DU DAG
# =============================================================================
@dag(dag_id='rakuten_create_dataset',
     default_args={
        'owner': 'rakuten-team',
        'start_date': timezone.datetime(2025, 1, 1),
        'retries': 0,  # Pas de retry pour le debug, on veut voir l'erreur tout de suite
         },
     catchup=False,
     tags=['mlops', 'rakuten', 'docker'])
def create_dataset_dag():
    start = start_pipeline_task()

    # --- Étape 1 : Création du dataset ---
    dataset = dataset_task()

    # --- Étape 2 : Preprocessing ---
    preprocess = preprocess_task()

    end = end_pipeline_task()

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================

    start >> dataset >> preprocess >> end

dag = create_dataset_dag()
