import os

from pathlib import Path
from minio import Minio
from minio.error import S3Error


class BucketNotFoundError(Exception):
    """Exception levée lorsqu'un bucket MinIO n'existe pas."""

    def __init__(self, bucket_name, endpoint):
        message = f"Le bucket '{bucket_name}' est introuvable sur le serveur MinIO ({endpoint})."
        super().__init__(message)
        self.bucket_name = bucket_name
        self.endpoint = endpoint


def minio_bucket_connect(bucket_name: str):
    host = os.getenv("MINIO_HOST", "localhost")
    port = int(os.getenv("MINIO_PORT", "9000"))
    endpoint = f"{host}:{port}"
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")

    client = Minio(endpoint=endpoint,
                   access_key=access_key,
                   secret_key=secret_key,
                   secure=False)

    if client.bucket_exists(bucket_name):
        print(f"connexion au bucket {bucket_name} réussie")
        return client
    else:
        raise BucketNotFoundError(bucket_name, endpoint)


def push_data(bucket_name: str, local_path: str):
    """
    Envoie récursivement tout le contenu de local_path vers le bucket,
    en conservant exactement l'arborescence originale.
    """
    client = minio_bucket_connect(bucket_name)
    local_dir = Path(local_path)

    if not local_dir.exists():
        raise FileNotFoundError(f"❌ Dossier local introuvable : {local_path}")

    print(f"🚀 MIRROR LOCAL → S3 : {local_path} → bucket '{bucket_name}'")

    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            # object_name = chemin relatif complet dans le bucket
            object_name = str(file_path.relative_to(local_dir))

            try:
                client.fput_object(bucket_name, object_name, str(file_path))
                print(f"📤 UPLOAD : {object_name}")
            except S3Error as e:
                print(f"⚠️ Erreur upload {file_path}: {e}")


def pull_data(bucket_name: str, local_dest: str):
    """
    Télécharge récursivement tout le bucket dans local_dest,
    en conservant l'arborescence du bucket.
    """
    client = minio_bucket_connect(bucket_name)
    local_dir = Path(local_dest)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ MIRROR S3 → LOCAL : bucket '{bucket_name}' → {local_dest}")

    objects = client.list_objects(bucket_name, recursive=True)

    for obj in objects:
        # Chemin local complet à recréer
        local_file_path = local_dir / obj.object_name

        # Création auto des sous-dossiers
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            client.fget_object(bucket_name, obj.object_name, str(local_file_path))
            print(f"📥 DOWNLOAD : {obj.object_name}")
        except S3Error as e:
            print(f"⚠️ Erreur download {obj.object_name}: {e}")
