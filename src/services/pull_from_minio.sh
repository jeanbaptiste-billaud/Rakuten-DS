#!/bin/sh
set -e

# Vérifie qu’un argument (le dossier à synchroniser) est fourni
if [ -z "$1" ]; then
  echo "❌ Erreur : aucun nom de dossier fourni."
  echo "Usage : $0 <nom_dossier>"
  exit 1
fi

BUCKET_PATH="$1"

# Configurer l’alias MinIO
mc alias set myminio http://${MINIO_HOST}:${MINIO_PORT} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Créer le dossier local s’il n’existe pas
mkdir -p /mnt/data/${BUCKET_PATH}

echo "⬇️ Synchronisation du bucket myminio/${BUCKET_PATH} vers /mnt/data/${BUCKET_PATH}"
mc mirror --overwrite myminio/${BUCKET_PATH} /mnt/data/${BUCKET_PATH}

echo "✅ Synchronisation terminée : ${BUCKET_PATH}"
