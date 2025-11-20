#!/bin/bash
set -eu
# -e : stoppe en cas d’erreur
# -u : stoppe si une variable n’est pas définie

# Déterminer le chemin absolu du script (utile si exécuté depuis un autre dossier)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Charger le fichier .env (ici, il est dans le même dossier que docker-compose.yml)
ENV_PATH="$SCRIPT_DIR/../docker-compose/.env"

if [ ! -f "$ENV_PATH" ]; then
  echo "❌ Fichier .env introuvable à l'emplacement : $ENV_PATH"
  exit 1
fi

set -a  # export automatique
source "$ENV_PATH"
set +a

echo "🔗 Initialisation des remotes DVC avec MinIO à localhost:${MINIO_PORT}"

# Liste des buckets à initialiser
declare -A remotes=(
  [minio_raw]="raw"
  [minio_dataset]="dataset"
  [minio_preprocessed]="preprocessed"
)

# Création des remotes
for name in "${!remotes[@]}"; do
  bucket="${remotes[$name]}"

  echo "🪣 Création du remote '$name' → s3://$bucket"
  dvc remote add -f "$name" "s3://$bucket"
  dvc remote modify "$name" endpointurl "http://localhost:${MINIO_PORT}"
  dvc remote modify "$name" use_ssl false
  dvc remote modify "$name" --local access_key_id "$MINIO_ROOT_USER"
  dvc remote modify "$name" --local secret_access_key "$MINIO_ROOT_PASSWORD"
done

echo "✅ Configuration DVC terminée avec succès."
