import os
import shlex

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================

WORKDIR = os.getenv("WORKDIR", "/app")
INFISICAL_DOMAIN = os.getenv("INFISICAL_DOMAIN", "http://infisical:8080")
INFISICAL_ENV = os.getenv("INFISICAL_ENV", "dev")
INFISICAL_PROJECT_ID = os.getenv("INFISICAL_PROJECT_ID", "")

INFISICAL_SECRET_PATHS = {
    "data-ingestion-id": "/workloads/data-ingestion",
    "data-enrichment-id": "/workloads/data-enrichment",
    "data-preprocessing-id": "/workloads/data-preprocessing",
    "training-id": "/workloads/training",
    "model-evaluation-id": "/workloads/model-evaluation",
    "model-build-id": "/workloads/model-build",
    "backup-id": "/workloads/backup",
    "postgres-backup-id": "/workloads/postgres-backup",
    "dvc-publisher-id": "/workloads/dvc-publisher",
}


def _identity_token_env(identity: str) -> str:
    return f"INFISICAL_{identity.upper().replace('-', '_')}_TOKEN"


def infisical_runtime_env(identity: str) -> dict[str, str]:
    token_env = _identity_token_env(identity)
    return {
        "INFISICAL_DOMAIN": INFISICAL_DOMAIN,
        "INFISICAL_ENV": INFISICAL_ENV,
        "INFISICAL_PROJECT_ID": INFISICAL_PROJECT_ID,
        "INFISICAL_IDENTITY": identity,
        "INFISICAL_SECRET_PATH": INFISICAL_SECRET_PATHS[identity],
        "INFISICAL_TOKEN": os.getenv(token_env, ""),
    }


def infisical_run_command(script: str, identity: str):
    quoted_script = shlex.quote(script)
    return [
        "sh",
        "-lc",
        f"""
        set -e
        : "${{INFISICAL_TOKEN:?missing Infisical token for {identity}}}"
        : "${{INFISICAL_PROJECT_ID:?missing Infisical project id}}"
        exec infisical run \
          --silent \
          --domain "$INFISICAL_DOMAIN" \
          --token "$INFISICAL_TOKEN" \
          --projectId "$INFISICAL_PROJECT_ID" \
          --env "$INFISICAL_ENV" \
          --path "$INFISICAL_SECRET_PATH" \
          -- sh -lc {quoted_script}
        """.strip(),
    ]


def docker_common_args(identity: str | None = None, extra_environment: dict[str, str] | None = None):
    environment = {
        "WORKDIR": WORKDIR,
        "MINIO_HOST": "minio",
        "MINIO_PORT": "9000",
        "MLFLOW_TRACKING_URI": "http://mlflow-server:5000",
        "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    }

    if identity:
        environment.update(infisical_runtime_env(identity))

    if extra_environment:
        environment.update(extra_environment)

    return {
        "api_version": "auto",
        "auto_remove": "success",
        "network_mode": "mlflow-network",
        "environment": environment,
    }


# =============================================================================
# 🛠️ DÉFINITION DES TASK
# =============================================================================

def start_pipeline_task():
    return BashOperator(
        task_id='start_pipeline',
        bash_command='echo "🚀 Démarrage du pipeline Rakuten"'
    )


def end_pipeline_task():
    return BashOperator(
        task_id='end_pipeline',
        bash_command='echo "✅ Pipeline terminé avec succès"'
    )


def preprocess_task():
    identity = "data-preprocessing-id"
    common_args = docker_common_args(identity)
    script = """
        echo "⬇️ Downloading inputs..." &&
        python /src/utils/sync_bucket.py dataset --mode pull &&

        echo "⚙️ Processing Preprocessing..." &&
        python /src/preprocessing.py &&

        echo "⬆️ Uploading results..." &&
        python /src/utils/sync_bucket.py preprocessed --mode push
    """
    return DockerOperator(
        task_id='preprocessing',
        image='jbbillaud/rakuten:spacy-v3.8.11',
        command=infisical_run_command(script, identity),
        doc_md="""
        ### 🐳 Docker task
        - Lance un conteneur Ubuntu
        - Affiche `hello`
        - Sert de test
        """,
        **common_args
    )


def volume_backup_task(data_volume: str):
    common_args = docker_common_args()
    backup_dir = f"{WORKDIR}/dvc_data/Rakuten-DS/data"
    backup_file = f"{backup_dir}/{data_volume}.tar.gz"

    return DockerOperator(
        task_id=f'backup_{data_volume}_volume',
        image='jbbillaud/rakuten:dvc-v3.66.1',
        mounts=[
            Mount(source=data_volume, target=os.path.join(WORKDIR, data_volume), type="volume"),
            Mount(source="dvc_data", target=os.path.join(WORKDIR, "dvc_data"), type="volume")
        ],
        command=f"""
                    sh -c "
                    set -e ;
                    echo 'backup {data_volume}...' ;
                    mkdir -p '{backup_dir}' ;
                    tar -czf '{backup_file}' -C '{WORKDIR}' '{data_volume}' ;
                    ls -lh '{backup_file}'
                    "
                """,
        doc_md="""
                ### 🐳 Docker task
                - Backup d'un volume Docker
                - Archive le volume dans `dvc_data`
                """,
        **common_args,

    )

def pg_dump_task(db_name: str):
    identity = "postgres-backup-id"
    common_args = docker_common_args(identity)
    out_dir = f"{WORKDIR}/dvc_data/Rakuten-DS/data/pg_backups"
    out_file = f"{out_dir}/{db_name}.dump"
    script = f"""
        set -e
        : "${{POSTGRES_DUMP_USER:?missing POSTGRES_DUMP_USER}}"
        : "${{POSTGRES_DUMP_PASSWORD:?missing POSTGRES_DUMP_PASSWORD}}"
        uri="postgresql://${{POSTGRES_DUMP_USER}}:${{POSTGRES_DUMP_PASSWORD}}@postgres:5432/{db_name}"
        echo 'dump db={db_name}...'
        mkdir -p '{out_dir}'
        pg_dump -Fc -C "$uri" -f '{out_file}'
        ls -lh '{out_file}'
    """

    return DockerOperator(
        task_id=f"backup_{db_name}",
        image="postgres:17",  # contient pg_dump/pg_restore
        mounts=[
            Mount(source="dvc_data", target=os.path.join(WORKDIR, "dvc_data"), type="volume"),
        ],
        command=infisical_run_command(script, identity),
        **common_args,
    )
