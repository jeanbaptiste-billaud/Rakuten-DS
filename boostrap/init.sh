#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

REPO_OWNER="jeanbaptiste-billaud"
REPO_NAME="Rakuten-DS"
REPO_HTTPS="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"

# -----------------------------
# Helpers
# -----------------------------
prompt_secret() {
  local __var_name="$1"
  local __prompt="$2"
  local __val
  read -r -s -p "$__prompt" __val
  echo
  printf -v "$__var_name" '%s' "$__val"
}

prompt() {
  local __var_name="$1"
  local __prompt="$2"
  local __default="${3:-}"
  local __val
  if [[ -n "$__default" ]]; then
    read -r -p "$__prompt [$__default]: " __val
    __val="${__val:-$__default}"
  else
    read -r -p "$__prompt: " __val
  fi
  printf -v "$__var_name" '%s' "$__val"
}

# -----------------------------
# 0) Inputs (host side)
# -----------------------------
echo "=== Bootstrap init ==="

prompt_secret GITHUB_TOKEN "GitHub token (fine-grained PAT) pour cloner le repo privé: "
prompt DAGSHUB_USER "DagsHub user" "jeanbaptiste-billaud"
prompt_secret DAGSHUB_PASSWORD "DagsHub token/password (DVC auth basic): "

echo
echo "=== Phase 1: DVC (clone + config remote + dvc pull) ==="

docker compose -f "$COMPOSE_FILE" run --rm \
  -e GITHUB_TOKEN="${GITHUB_TOKEN}" \
  -e DAGSHUB_USER="${DAGSHUB_USER}" \
  -e DAGSHUB_PASSWORD="${DAGSHUB_PASSWORD}" \
  dvc bash -lc '
    set -euo pipefail
    cd "$HOME/dvc_data"

    if [[ ! -d "'"$REPO_NAME"'" ]]; then
      echo "[DVC] Cloning repo (private) ..."
      # Clone avec token via HTTPS
      git clone "https://${GITHUB_TOKEN}@github.com/'"$REPO_OWNER"'/'"$REPO_NAME"'.git" "'"$REPO_NAME"'"
    else
      echo "[DVC] Repo already present, skipping clone."
    fi

    cd "'"$REPO_NAME"'"

    echo "[DVC] Configuring DVC remote origin (local auth)..."
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user "${DAGSHUB_USER}"
    dvc remote modify origin --local password "${DAGSHUB_PASSWORD}"

    echo "[DVC] Pulling DVC tracked data..."
    sed -i "s/\r$//" .dvc/dvc_pull_list.txt
    grep -vE "^\s*($|#)" .dvc/dvc_pull_list.txt | xargs -d "\n" dvc pull

    echo "[DVC] Done."

    echo "postgres db restoration"
    tar -xzf data/mlflow_db.tar.gz -C "$HOME/pgdata"
  '
echo
echo "=== Phase 2: MinIO (start + bucket population via /src/minio_init.sh) ==="

# 1) Start MinIO and helper client (bucket create etc.)
docker compose -f "$COMPOSE_FILE" up -d minio-client

# 2) Execute your init script INSIDE minio container image (as you requested)
# We override the command to run the script once and exit.
docker compose -f "$COMPOSE_FILE" run --rm \
  --entrypoint /bin/sh \
  minio -lc "/src/minio_init.sh"

echo
echo "=== Bootstrap terminé ==="
