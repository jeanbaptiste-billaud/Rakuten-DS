import os

from minio import Minio


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
                   secret_key=secret_key)

    if client.bucket_exists(bucket_name):
        print(f"connexion au bucket {bucket_name} réussie")
        return client
    else:
        raise BucketNotFoundError(bucket_name, endpoint)

def pull_data(bucket_name: str):
    client = minio_bucket_connect(bucket_name)

def push_data(bucket_name: str):
    client = minio_bucket_connect(bucket_name)