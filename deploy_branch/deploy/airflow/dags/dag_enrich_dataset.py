from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import dag, timezone, task

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
def enrich_task():
    identity = "data-enrichment-id"
    common_args = docker_common_args(identity)
    script = """
        echo "⬇️ Downloading inputs..." &&
        python /src/utils/sync_bucket.py raw --mode pull &&
        python /src/utils/sync_bucket.py dataset --mode pull &&

        echo "⚙️ Processing Enrich..." &&
        python /src/enrich_raw_dataset.py &&

        echo "⬆️ Uploading results..." &&
        python /src/utils/sync_bucket.py dataset --mode push
    """
    return DockerOperator(
        task_id='enrich_dataset',
        image="jbbillaud/rakuten:spacy-v3.8.11",
        command=infisical_run_command(script, identity),
        doc_md="""
        ### 🐳 Docker task
        - Lance un conteneur Ubuntu
        - Affiche `hello`
        - Sert de test
        """,
        **common_args
    )


# =============================================================================
# 🚀 DÉFINITION DU DAG
# =============================================================================
@dag(dag_id='rakuten_enrich_dataset',
     default_args={
            'owner': 'rakuten-team',
            'start_date': timezone.datetime(2025, 1, 1),
            'retries': 0,  # Pas de retry pour le debug, on veut voir l'erreur tout de suite
            },
     catchup=False,
     tags=['mlops', 'rakuten', 'docker'])
def enrich_dataset_dag():
    start = start_pipeline_task()

    # --- Étape 1 : Enrichissement ---
    enrich = enrich_task()

    # --- Étape 2 : Preprocessing ---
    preprocess = preprocess_task()

    end = end_pipeline_task()

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================

    start >> enrich >> preprocess >> end

dag = enrich_dataset_dag()
