import os

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================

MINIO_USER = os.getenv("MINIO_ROOT_USER", "minio")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minio123")
WORKDIR = os.getenv("WORKDIR", "/app")


def docker_common_args():
    return {
        "api_version": "auto",
        "auto_remove": "success",
        "network_mode": "mlflow-network",
        "environment": {
            "WORKDIR": WORKDIR,
            # Config MLflow & MinIO
            "MINIO_HOST": "minio",
            "MINIO_PORT": "9000",
            "MLFLOW_TRACKING_URI": "http://mlflow-server:5000",
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
            "MINIO_ACCESS_KEY": MINIO_USER,
            "MINIO_SECRET_KEY": MINIO_PASS,
            "AWS_ACCESS_KEY_ID": MINIO_USER,
            "AWS_SECRET_ACCESS_KEY": MINIO_PASS,
        }
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
    common_args = docker_common_args()
    return DockerOperator(
        task_id='preprocessing',
        image='jbbillaud/rakuten:spacy-v3.8.11',
        command="""sh -c '
            echo "⬇️ Downloading inputs..." &&
            python /src/utils/sync_bucket.py dataset --mode pull &&

            echo "⚙️ Processing Preprocessing..." &&
            python /src/preprocessing.py &&

            echo "⬆️ Uploading results..." &&
            python /src/utils/sync_bucket.py preprocessed --mode push
        '""",
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

def pg_dump_task(db_name: str, user: str, password: str):
    common_args = docker_common_args()
    out_dir = f"{WORKDIR}/dvc_data/Rakuten-DS/data/pg_backups"
    out_file = f"{out_dir}/{db_name}.dump"
    uri = f"postgresql://{user}:{password}@postgres:5432/{db_name}"


    return DockerOperator(
        task_id=f"backup_{db_name}",
        image="postgres:17",  # contient pg_dump/pg_restore
        mounts=[
            Mount(source="dvc_data", target=os.path.join(WORKDIR, "dvc_data"), type="volume"),
        ],
        command=f"""
        sh -c "set -e \
        && echo 'dump db={db_name}...' \
        && mkdir -p '{out_dir}' \
        && pg_dump -Fc -C '{uri}' -f '{out_file}' \
        && ls -lh '{out_file}'"
        """,
        **common_args,
    )
