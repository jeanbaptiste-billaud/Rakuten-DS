import os
from datetime import datetime

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import timezone, dag
from docker.types import Mount

from common_task import start_pipeline_task, end_pipeline_task, volume_backup_task, docker_common_args, pg_dump_task

# =============================================================================
# 🛠️ DÉFINITION DES TASK
# =============================================================================
WORKDIR = os.getenv("WORKDIR", "/app")


def minio_backup_task():
    common_args = docker_common_args()
    common_args["environment"]["MINIO_MLFLOW_BUCKET"] = os.getenv("MINIO_MLFLOW_BUCKET", "mlflow-artifacts")
    return DockerOperator(
        task_id='backup_minio_volume',
        image="jbbillaud/rakuten:minio-client-RELEASE.2025-08-13T08-35-41Z",
        mounts=[Mount(source="dvc_data", target="/dvc_data", type="volume")],
        command="/src/minio_backup.sh > /proc/1/fd/1 2>&1",
        doc_md="""
        ### 🐳 Docker task
        - Lance un conteneur minio-client
        - Execute le script de sauvegarde des buckets vers dvc_data
        """,
        **common_args
    )


def commit_task():
    common_args = docker_common_args()
    repo = f"{WORKDIR}/dvc_data/Rakuten-DS"
    return DockerOperator(
        task_id='git_dvc_commit',
        image='jbbillaud/rakuten:dvc-v3.66.1',
        mounts=[Mount(source="dvc_data", target=os.path.join(WORKDIR, "dvc_data"), type="volume")],
        command=f"""
        sh -c "set -e
        cd '{repo}'

        git config --global --add safe.directory '{repo}'
        git config --global user.email 'airflow@local'
        git config --global user.name 'airflow'
        
        # Lire la liste de tracking (en supprimant CRLF + vides + commentaires)
        sed -i 's/\\r$//' .dvc/dvc_tracking_list.txt
        TRACKED_PATHS=$(grep -vE '^\\s*($|#)' .dvc/dvc_tracking_list.txt || true)

        if [ -z \"$TRACKED_PATHS\" ]; then
          echo '[DVC] tracking list empty -> nothing to do'
        else
          echo '[DVC] Updating .dvc pointers via dvc add'
          # Met à jour les fichiers .dvc (hash) si les contenus ont changé
          printf '%s\\n' \"$TRACKED_PATHS\" | xargs -d '\\n' -r dvc add
        fi

        # Stage uniquement les fichiers DVC + la liste (si tu veux la versionner)
        git add -A '*.dvc' .dvc/dvc_tracking_list.txt .gitignore || true

        # Commit/push git seulement si quelque chose a changé
        if git diff --cached --quiet; then
          echo 'nothing to commit'
        else
          git commit -m 'automatique commit from airflow'
          git push origin dvc
        fi
        
        # Push data vers le remote DVC
        if [ -n \"$TRACKED_PATHS\" ]; then
          echo '[DVC] Pushing data to remote'
          dvc push
        else
          echo '[DVC] No tracked paths -> skip dvc push'
        fi
        "
        """,
        doc_md="""
        ### 🐳 Docker task
        - lit `.dvc/dvc_tracking_list.txt` (CRLF safe)
        - met à jour les `.dvc` via `dvc add` si les data ont changé
        - commit/push git uniquement si nécessaire
        - push des données via `dvc push`
        """,
        **common_args,
    )


pguser = "{{ conn.postgres_default.login }}"
pgpwd = "{{ conn.postgres_default.password }}"

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

    backup_mlflow_db = pg_dump_task(os.getenv("MLFLOW_DB", "mlflow_db"), "mlflow", "mlflow")
    backup_airflow_db = pg_dump_task(os.getenv("MLFLOW_DB", "airflow_db"), "mlflow", "mlflow")
    backup_logs_and_reports = volume_backup_task("logs_and_reports")
    backup_minio_volume = minio_backup_task()
    # commit = commit_task()

    # end = end_pipeline_task()

    # =========================================================================
    # 🔗 ORCHESTRATION
    # =========================================================================

    # start >> [backup_minio_volume, backup_logs_and_reports] >> backup_mlflow_db >> backup_airflow_db >> commit >> end
    start >> [backup_minio_volume, backup_logs_and_reports] >> backup_mlflow_db >> backup_airflow_db


dag = backup_pipeline_dag()