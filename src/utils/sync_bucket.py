import argparse
import os
from src.utils.minio_utils import push_data, pull_data  # adapte l’import selon ton projet


def main():
    parser = argparse.ArgumentParser(description="Sync d’un dossier avec un bucket MinIO.")
    parser.add_argument(
        "folder",
        help="Nom du dossier local (et nom du bucket MinIO)"
    )
    parser.add_argument(
        "--mode",
        choices=["push", "pull"],
        required=True,
        help="push = uploader vers MinIO, pull = télécharger depuis MinIO"
    )

    args = parser.parse_args()

    bucket_name = args.folder
    local_folder = os.path.join("/workspace/data", args.folder)  # puisque dossier = bucket

    if args.mode == "push":
        print(f"📤 PUSH : {local_folder} → bucket `{bucket_name}`")
        os.makedirs(local_folder, exist_ok=True)
        push_data(bucket_name, local_folder)
    else:
        print(f"📥 PULL : bucket `{bucket_name}` → {local_folder}")
        pull_data(bucket_name, local_folder)


if __name__ == "__main__":
    main()
