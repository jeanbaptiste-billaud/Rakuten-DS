#!/usr/bin/env python3
"""Back up MinIO buckets into the DVC volume with the MinIO Python SDK."""

from __future__ import annotations

import os
from minio.error import S3Error

from src.minio.minio_common import create_client, data_path


def main() -> None:
    client = create_client()
    destination_root = data_path("MINIO_BACKUP_DATA_PATH")
    mlflow_bucket = os.getenv("MINIO_MLFLOW_BUCKET", "mlflow-artifacts")

    for bucket in ("dataset", "preprocessed", mlflow_bucket):
        destination = destination_root / bucket
        print(f"Traitement du bucket : {bucket}")
        count = 0
        try:
            for item in client.list_objects(bucket, recursive=True):
                if item.is_dir:
                    continue
                target = destination / item.object_name
                target.parent.mkdir(parents=True, exist_ok=True)
                client.fget_object(bucket, item.object_name, str(target))
                count += 1
        except S3Error as exc:
            raise RuntimeError(f"Échec de sauvegarde du bucket {bucket}: {exc}") from exc
        print(f"{bucket}: {count} objet(s) sauvegardé(s) dans {destination}")

    print("Toutes les sauvegardes MinIO sont terminées.")


if __name__ == "__main__":
    main()
