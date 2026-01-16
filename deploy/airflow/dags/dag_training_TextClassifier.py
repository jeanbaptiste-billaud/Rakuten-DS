import os
import textwrap
from datetime import datetime

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import timezone, dag
from docker.types import Mount

from common_task import start_pipeline_task, end_pipeline_task, docker_common_args

# =============================================================================
# 🛠️ DÉFINITION DES TASK
# =============================================================================
WORKDIR = os.getenv("WORKDIR", "/app")


def lineage_task():
    common_args = docker_common_args()
    return DockerOperator(
        task_id="generate_data_lineage_json_file",
        image="jbbillaud/rakuten:dvc-v3.66.1",
        mounts=[Mount(source="dvc_data", target=os.path.join(WORKDIR, "dvc_data"), type="volume"),
                Mount(source="airflow_vol", target=WORKDIR, type="volume")],
        user=f"{os.getenv('AIRFLOW_UID', 5000)}:0",
        command=[
            "sh", "-lc", f"""
            set -e
            python /src/generate_data_lineage.py \
              --project-root {WORKDIR}/dvc_data/Rakuten-DS \
              --rev HEAD \
              --out {WORKDIR}/lineage/
            """.strip()
        ],
        **common_args
    )


def training_task():
    common_args = docker_common_args()
    script = textwrap.dedent(f"""\
        echo "⬇️ Downloading preprocessed data..."
        python /src/utils/sync_bucket.py preprocessed --mode pull

        echo "🧠 Training model..."
        python /src/train_text_model_with_drift.py
        
        cp -r /tmp/* {WORKDIR}/training_exports
    """)

    return DockerOperator(
        task_id="train_model",
        image="jbbillaud/rakuten:sklearn-v1.8.0",
        mounts=[Mount(source="airflow_vol", target=WORKDIR, type="volume")],
        command=["sh", "-lc", script],
        **common_args,
    )


def build_model_task():
    common_args = docker_common_args()
    return DockerOperator(
        task_id="build_model",
        image="jbbillaud/rakuten:bentoml-v1.4.33",
        user=f"{os.getenv('AIRFLOW_UID', 5000)}:{os.getenv('DOCKER_GID', 1001)}",
        command=["sh", "-lc", """
            set -e
            . /venv/bin/activate
            sh /model_serving/model_building.sh
            """],
        mounts=[Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
                Mount(source="airflow_vol", target=WORKDIR, type="volume")],
        **common_args,
    )


def stop_model_task():
    return BashOperator(
        task_id="stop_model_serving",
        bash_command="""
        set -e
        docker compose -f /opt/airflow/compose/docker-compose.yml stop model-serving
        """,
    )


def start_model_task():
    return BashOperator(
        task_id="start_model_serving",
        bash_command="""
        set -e
        docker compose -f /opt/airflow/compose/docker-compose.yml up -d model-serving
        """,
    )


# =============================================================================
# 🚀 DÉFINITION DU DAG
# =============================================================================

default_args = {
    'owner': 'rakuten-team',
    'start_date': timezone.datetime(2025, 1, 1),
    'retries': 0,  # Pas de retry pour le debug, on veut voir l'erreur tout de suite
}


@dag(
    dag_id="training_pipeline",
    default_args=default_args,
    catchup=False,
    tags=["mlops", "rakuten", "docker", "training"],
    start_date=datetime(2024, 1, 1),  # ajuste si tu as déjà un start_date ailleurs
)
def training_pipeline_dag():
    start = start_pipeline_task()

    data_lineage = lineage_task()
    training = training_task()

    stop_model = stop_model_task()
    build = build_model_task()
    start_model = start_model_task()

    end = end_pipeline_task()

    # Orchestration
    start >> data_lineage >> training >> stop_model >> build >> start_model >> end


dag = training_pipeline_dag()
