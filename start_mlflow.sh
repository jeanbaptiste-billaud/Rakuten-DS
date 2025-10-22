#!/bin/bash

set -e  # stoppe le script si une commande échoue

CONTAINER_NAME="mlflow-server"
IMAGE_NAME="ghcr.io/mlflow/mlflow:v3.5.0"

# Vérifie si un conteneur portant ce nom existe déjà
if [ "$(docker ps -a -q -f name=^${CONTAINER_NAME}$)" ]; then
    # Si le conteneur tourne déjà
    if [ "$(docker ps -q -f name=^${CONTAINER_NAME}$)" ]; then
        echo "🚀 MLflow est déjà en cours d'exécution."
        exit 0
    else
        echo "🧹 Un ancien conteneur MLflow existe, on le supprime..."
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
    fi
fi

## Crée les dossiers si nécessaire
#mkdir -p mlflow/mlruns
#touch mlflow/mlflow.db

echo "🚀 Lancement du serveur MLflow..."
docker run  --mount type=bind,src=$(pwd)/mlflow,dst=/mlflow,bind-propagation=shared -d \
  --name "$CONTAINER_NAME" \
  -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db \
  -e MLFLOW_ARTIFACT_ROOT=/mlflow/mlruns \
  "$IMAGE_NAME" \
  mlflow server \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root http://localhost:5000/mlflow/mlruns \
  --host 0.0.0.0 \
  --port 5000

echo "✅ Serveur MLflow démarré sur http://localhost:5000"
