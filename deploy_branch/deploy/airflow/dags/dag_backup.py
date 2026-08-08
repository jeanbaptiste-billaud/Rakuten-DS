import os
from datetime import datetime

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import timezone, dag
from docker.types import Mount

from common_task import (
    docker_common_args,
    end_pipeline_task,
    infisical_run_command,
    infisical_run_argv,
    security_mounts,
    pg_dump_task,
    start_pipeline_task,
    volume_backup_task,
)

# =============================================================================
# 🛠️ DÉFINITION DES TASK
# =============================================================================
WORKDIR = os.getenv("WORKDIR", "/app")


def minio_backup_task():
    identity = "backup-id"
    common_args = docker_common_args(
        identity,
        extra_environment={"MINIO_MLFLOW_BUCKET": os.getenv("MINIO_MLFLOW_BUCKET", "mlflow-artifacts")},
    )
    return DockerOperator(
        task_id='backup_minio_volume',
        image="jbbillaud/rakuten:minio-client-latest",
        user="0:0",
        mounts=security_mounts(Mount(source="dvc_data", target="/dvc_data", type="volume")),
        command=infisical_run_argv(["python", "-m", "src.minio.minio_backup"], identity),
        doc_md="""
        ### 🐳 Docker task
        - Lance un conteneur minio-client
        - Execute le script de sauvegarde des buckets vers dvc_data
        """,
        **common_args
    )


def commit_task():
    identity = "dvc-publisher-id"
    common_args = docker_common_args(identity)
    repo = f"{WORKDIR}/dvc_data/Rakuten-DS"
    return DockerOperator(
        task_id='git_dvc_commit',
        image='jbbillaud/rakuten:dvc-v3.66.1',
        mounts=security_mounts(
            Mount(source="dvc_data", target=os.path.join(WORKDIR, "dvc_data"), type="volume")
        ),
        command=infisical_run_command(f"""
        sh -c "set -e
        cd '{repo}'

        git config --global --add safe.directory '{repo}'
        git config --global user.email 'airflow@local'
        git config --global user.name 'airflow'
        
        # Lire la liste de tracking (en supprimant CRLF + vides + commentaires)
        sed -i 's/\\r$//' .dvc/dvc_tracking_list.txt
        TRACKED_PATHS=$(grep -vE '^\\s*($|#)' .dvc/dvc_tracking_list.txt | sed 's/\\.dvc$//' | sort -u || true)

        if [ -z "$TRACKED_PATHS" ]; then
          echo '[DVC] tracking list empty -> nothing to do'
        else
          echo '[DVC] Updating DVC pointers via dvc add'
          printf '%s\n' "$TRACKED_PATHS" | xargs -d '\n' -r dvc add
        fi

        # Stage uniquement les fichiers qui ont été modifiés
        git add -A || true

        # Commit/push git seulement si quelque chose a changé
        if git diff --cached --quiet; then
          echo '[GIT] nothing to commit'
        else
          git commit -m 'automatique commit from airflow'
        fi
        
        git push origin dvc
        dvc push
        "
        """, identity),
        doc_md="""
        ### 🐳 Docker task
        - lit `.dvc/dvc_tracking_list.txt` (CRLF safe)
        - met à jour les fichiers suivi par dvc via `dvc add` si les datas ont changé
        - commit/push git uniquement si nécessaire
        - push des données via `dvc push`
        """,
        **common_args,
    )


# =============================================================================
# 🚀 DÉFINITION DU DAG
# =============================================================================

default_args = {
    'owner': 'rakuten-team',
    'start_date': timezone.datetime(2025, 1, 1),
    'retries': 0,  # Pas de retry pour le debug, on veut voir l'erreur tout de suite
}


@dag(dag_id='rakuten_backup_pipeline',
     default_args=default_args,
     catchup=False,
     tags=['mlops', 'rakuten', 'docker'],
     start_date=datetime(2024, 1, 1),
     )
def backup_pipeline_dag():
    start = start_pipeline_task()

    backup_mlflow_db = pg_dump_task(os.getenv("MLFLOW_DB", "mlflow_db"))
    backup_airflow_db = pg_dump_task(os.getenv("AIRFLOW_DB", "airflow_db"))
    backup_logs_and_reports = volume_backup_task("logs_and_reports")
    backup_minio_volume = minio_backup_task()
    commit = commit_task()

    end = end_pipeline_task()

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================

    start >> [backup_minio_volume, backup_logs_and_reports] >> backup_mlflow_db >> backup_airflow_db >> commit >> end
    # start >> [backup_minio_volume, backup_logs_and_reports] >> backup_mlflow_db >> backup_airflow_db


dag = backup_pipeline_dag()
