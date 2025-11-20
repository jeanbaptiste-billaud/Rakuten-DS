#!/bin/bash
set -eu
# 🟢 PUSH local → MinIO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Push des données DVC vers MinIO)"

# Vérifier que DVC est initialisé
if [ ! -d ".dvc" ]; then
  echo "❌ DVC n'est pas initialisé dans ce répertoire."
  exit 1
fi

# Liste des buckets à initialiser
declare -A remotes=(
  [minio_raw]="raw"
  [minio_dataset]="dataset"
  [minio_preprocessed]="preprocessed"
)

# Envoi des données vers le bucket correspondant
for remote_name in "${!remotes[@]}"; do

  data_name="${remotes[$remote_name]}"
  if [ "$data_name" = "raw" ]; then
    dvc_file_path="$SCRIPT_DIR/../data/raw/all_raw_data.csv.dvc"
  else
    dvc_file_path="$SCRIPT_DIR/../data/${data_name}.dvc"
  fi

  dvc push -r $remote_name $dvc_file_path
done

# Effectuer le push vers le remote par défaut (ex: minio_dataset)
echo "📦 DVC push en cours..."
dvc push

echo "✅ Données envoyées vers MinIO avec succès."
