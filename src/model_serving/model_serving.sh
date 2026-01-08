#!/bin/bash
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# 1) sauvegarde du run_id
cat <<EOF > provenance.txt
mlflow_run_id=${MLFLOW_RUN_ID}
build_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

# 2) import du modèle dans le store bentoml à partir du run_id mlflow
python build_step.py

# 3) Construction d'un bento labellisé à partir du modèle
BENTO_TAG=$(bentoml build \
  --label "mlflow.run_id=${MLFLOW_RUN_ID}" \
  --label "mlflow.model_uri=${MODEL_URI}" \
  --output tag)

# 4) containerize avec label Docker + tag image parlant
docker_repo="jbbillaud/rakuten"
image_name="text-model"
docker_tag="${docker_repo}:${image_name}-run-${MLFLOW_RUN_ID}" # ou short/8 chars si tu veux

bentoml containerize "$BENTO_TAG" \
  -t "$docker_tag" \
  --label "io.mlflow.run_id=${MLFLOW_RUN_ID}" \
  --label "io.mlflow.model_uri=${MODEL_URI}"

# 5) création d'une copie "latest" de l'image
docker image tag "${docker_tag}" "${docker_repo}:${image_name}-latest"
