import os
from minio_utils import push_data

data_path = "/mnt/data"
folder_list = ["raw", "dataset", "preprocessed"]

print("🚀 Sync LOCAL → S3 (mirroring)…")

for folder in folder_list:
    local_path = os.path.join(data_path, folder)
    push_data(bucket_name=folder, local_path=local_path)

print("✅ Mirroring LOCAL → S3 terminé.")
