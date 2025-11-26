#!/bin/sh

# Désactivation du mode stop-on-error
# car mc mirror et mc cp retournent souvent un code non zéro.
set +e

echo "🔗 Configuration de l'alias MinIO..."
mc alias set myminio http://${MINIO_HOST}:${MINIO_PORT} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

echo "🪣 Création des buckets (ignore si existent déjà)..."
mc mb --ignore-existing myminio/${MINIO_MLFLOW_BUCKET:-mlflow}
mc mb --ignore-existing myminio/raw
mc mb --ignore-existing myminio/dataset
mc mb --ignore-existing myminio/preprocessed

echo "📤 Upload du fichier RAW (ignore si absent)..."
if [ -f /mnt/data/raw/all_raw_data.csv ]; then
    mc cp /mnt/data/raw/all_raw_data.csv myminio/raw/all_raw_data.csv
fi

echo "🔄 Mirroring dataset..."
if [ -d /mnt/data/dataset ]; then
    mc mirror --overwrite /mnt/data/dataset myminio/dataset
fi

echo "🔄 Mirroring preprocessed..."
if [ -d /mnt/data/preprocessed ]; then
    mc mirror --overwrite /mnt/data/preprocessed myminio/preprocessed
fi

echo "✅ Initialisation MinIO terminée."
