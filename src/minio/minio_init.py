#!/usr/bin/env python3
"""Create and populate MinIO buckets with the MinIO Python SDK."""

from __future__ import annotations

import os
from pathlib import Path

from src.minio.minio_common import create_client, data_path, ensure_bucket


def upload_tree(client, bucket: str, source: Path) -> int:
    if not source.is_dir():
        print(f"Répertoire absent, synchronisation ignorée : {source}")
        return 0
    count = 0
    for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
        object_name = file_path.relative_to(source).as_posix()
        client.fput_object(bucket, object_name, str(file_path))
        count += 1
    return count


def main() -> None:
    client = create_client(root_credentials=True)
    source_root = data_path("MINIO_INIT_DATA_PATH")
    mlflow_bucket = os.getenv("MINIO_MLFLOW_BUCKET", "mlflow-artifacts")

    for bucket in ("raw", "dataset", "preprocessed", mlflow_bucket):
        ensure_bucket(client, bucket)
        count = upload_tree(client, bucket, source_root / bucket)
        print(f"{bucket}: {count} objet(s) envoyé(s)")

    print("Initialisation MinIO terminée.")


if __name__ == "__main__":
    main()
