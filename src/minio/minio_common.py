"""Shared MinIO SDK helpers for shellless hardened containers."""

from __future__ import annotations

import os
from pathlib import Path

from minio import Minio


DEFAULT_DATA_PATH = Path("/dvc_data/Rakuten-DS/data")


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_client(*, root_credentials: bool = False) -> Minio:
    prefix = "MINIO_ROOT" if root_credentials else "MINIO"
    access_name = f"{prefix}_USER" if root_credentials else "MINIO_ACCESS_KEY"
    secret_name = f"{prefix}_PASSWORD" if root_credentials else "MINIO_SECRET_KEY"
    endpoint = f"{os.getenv('MINIO_HOST', 'minio')}:{os.getenv('MINIO_PORT', '9000')}"
    return Minio(
        endpoint,
        access_key=required_env(access_name),
        secret_key=required_env(secret_name),
        secure=env_bool("MINIO_SECURE"),
    )


def data_path(variable: str) -> Path:
    return Path(os.getenv(variable, str(DEFAULT_DATA_PATH)))


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Bucket créé : {bucket}")
    else:
        print(f"Bucket déjà présent : {bucket}")
