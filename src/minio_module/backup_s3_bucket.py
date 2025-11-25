import os
from minio_utils import pull_data

data_path = "test" #"/mnt/data"
folder_list = ["dataset", "preprocessed"]

print("⬇️ Sync S3 → LOCAL (mirroring)…")

for folder in folder_list:
    local_dest = os.path.join(data_path, folder)
    pull_data(bucket_name=folder, local_dest=local_dest)

print("✅ Mirroring S3 → LOCAL terminé.")
