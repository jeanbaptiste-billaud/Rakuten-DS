import os
import textwrap
from datetime import datetime

import requests
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.sdk import timezone, dag, task, task_group
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
    common_args["environment"]["DRIFT_DETECTOR_URL"] = os.getenv("DRIFT_DETECTOR_URL")
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


@task()
def get_model_run_id_task():
    response = requests.get("http://model-serving:8002/metadata", timeout=5)
    response.raise_for_status()

    payload = response.json()

    run_id = payload.get("mlflow_run_id")
    if not run_id:
        raise ValueError(f"mlflow_run_id missing in metadata payload: {payload}")
    else:
        return run_id


def model_comparison_task():
    common_args = docker_common_args()
    common_args["environment"]["MODEL_RUN_ID"] = "{{ ti.xcom_pull(task_ids='get_model_run_id_task') }}"
    return DockerOperator(
        task_id="model_comparison",
        image="jbbillaud/rakuten:sklearn-v1.8.0",
        mounts=[Mount(source="airflow_vol", target=WORKDIR, type="volume")],
        command=["sh", "-lc", "python /src/model_promotion_decision.py"],
        do_xcom_push=True,
        **common_args
    )


def keep_current_model_task():
    return BashOperator(
        task_id='keep_current_model',
        bash_command='echo "Conservation du modèle en cours d''utilisation"'
    )


@task_group(group_id="promote_model")
def build_new_model_task_gp():
    stop_model = stop_model_task()
    build = build_model_task()
    start_model = start_model_task()

    keep_current_model = keep_current_model_task()

    end = EmptyOperator(
        task_id="end_model_promote_gp",
        trigger_rule="none_failed_min_one_success"
    )

    [keep_current_model, stop_model >> build >> start_model] >> end


def branch_on_model_promotion(**context):
    result = context["ti"].xcom_pull(task_ids="model_comparison")

    # result est une string: "true" ou "false"
    if result == "true":
        return "promote_model.stop_model_serving"
    else:
        return "promote_model.keep_current_model"


# =============================================================================
# 🚀 DÉFINITION DU DAG
# =============================================================================
@dag(
    dag_id="rakuten_training_pipeline",
    default_args={
        'owner': 'rakuten-team',
        'start_date': timezone.datetime(2025, 1, 1),
        'retries': 0, },
    catchup=False,
    tags=["mlops", "rakuten", "docker", "training"],
    start_date=datetime(2024, 1, 1),  # ajuste si tu as déjà un start_date ailleurs
)
def training_pipeline_dag():
    start = start_pipeline_task()
    data_lineage = lineage_task()
    training = training_task()
    get_run_id = get_model_run_id_task()
    model_comparison = model_comparison_task()

    branch = BranchPythonOperator(
        task_id="branch_model_decision",
        python_callable=branch_on_model_promotion,
    )

    model_promotion = build_new_model_task_gp()
    end = end_pipeline_task()

    # Orchestration
    start >> data_lineage >> training >> get_run_id >> model_comparison >> branch >> model_promotion >> end


dag = training_pipeline_dag()
