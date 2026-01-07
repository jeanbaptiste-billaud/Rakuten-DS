#!/bin/bash
set -e

# Dossier où se trouve ce script
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

set -a
source "$SCRIPT_DIR/.env"
set +a

# On remonte d’un niveau (parent de docker/)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

REPO=jbbillaud/rakuten

build_push () {
  echo "build $2 image and push"
  docker buildx build -f $1 -t $REPO:$2 . --push
  echo
}

build_push ./docker/api-gateway/Dockerfile fastapi-v0.115.0
build_push ./docker/authentification/Dockerfile authentification
build_push ./docker/bentoml/Dockerfile bentoml-v1.4.30
build_push ./docker/drift-detector/Dockerfile evidently-v0.4.33
build_push ./docker/dvc/Dockerfile dvc-v3.64.0
build_push ./docker/minio-client/Dockerfile minio-client-RELEASE.2025-08-13T08-35-41Z
build_push ./docker/mlflow/Dockerfile mlflow-v3.6.0
build_push ./docker/spacy/Dockerfile spacy-v3.7.5
build_push ./docker/trainer/Dockerfile sklearn-v1.7.2
