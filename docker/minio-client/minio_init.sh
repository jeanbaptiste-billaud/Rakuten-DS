#!/bin/sh
set -euo pipefail

export MC_CONFIG_DIR=/tmp/.mc
rm -rf "$MC_CONFIG_DIR"
mkdir -p "$MC_CONFIG_DIR"

set +e

echo "🔗 Configuration de l'alias MinIO..."
mc alias set myminio http://${MINIO_HOST}:${MINIO_PORT} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

echo "🪣 Création du bucket raw et import du csv..."
DATA_PATH='/dvc_data/Rakuten-DS/data'
mc mb --ignore-existing myminio/raw
if [ -f ${DATA_PATH}/raw/all_raw_data.csv ]; then
    mc cp ${DATA_PATH}/raw/all_raw_data.csv myminio/raw/all_raw_data.csv
fi

# Liste des buckets à créer / synchroniser
BUCKETS_LIST=("dataset" "preprocessed" "${MINIO_MLFLOW_BUCKET}")

# Base locale contenant les données
echo ""
echo "Début de la création et synchronisation des buckets MinIO..."
echo "-----------------------------------------------------------"

for BUCKET in "${BUCKETS_LIST[@]}"; do
    echo ""
    echo "▶️  Traitement du bucket : ${BUCKET}"

    DESTINATION="myminio/${BUCKET}/"
    SOURCE="${DATA_PATH}/${BUCKET}/"

    echo "🪣 Création du bucket s'il n'existe pas..."
    mc mb --ignore-existing ${DESTINATION}

    echo "🔄 Mirroring du dossier local : ${SOURCE}"
    if [ -d "${SOURCE}" ]; then
        mc mirror --overwrite "${SOURCE}" "${DESTINATION}"
    else
        echo "⚠️  Le dossier local ${SOURCE} n'existe pas, on saute."
    fi
done

echo ""
echo "✅ Initialisation MinIO terminée."
