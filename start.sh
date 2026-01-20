#!/usr/bin/env bash
set -euo pipefail

# --- Helpers ---
die() { echo "❌ $*" >&2; exit 1; }

# Racine du repo (dossier où se trouve CE script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

BOOTSTRAP_DIR="${ROOT_DIR}/boostrap"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_DIR}/init.sh"

DEPLOY_COMPOSE_DIR="${ROOT_DIR}/deploy/compose"
DEPLOY_COMPOSE_FILE="${DEPLOY_COMPOSE_DIR}/docker-compose.yml"

# Toujours revenir à la racine à la fin
cleanup() {
  cd "${ROOT_DIR}" || true
}
trap cleanup EXIT

# --- Preflight checks ---
[[ -f "${BOOTSTRAP_SCRIPT}" ]] || die "Bootstrap script introuvable: ${BOOTSTRAP_SCRIPT}"
[[ -d "${DEPLOY_COMPOSE_DIR}" ]] || die "Dossier deploy/compose introuvable: ${DEPLOY_COMPOSE_DIR}"
[[ -f "${DEPLOY_COMPOSE_FILE}" ]] || die "docker-compose.yml introuvable: ${DEPLOY_COMPOSE_FILE}"

echo "=== Phase A: Bootstrap (DVC + restore DB + MinIO init) ==="
cd "${BOOTSTRAP_DIR}"
bash "${BOOTSTRAP_SCRIPT}"

echo
echo "=== Phase B: Déploiement (deploy/compose) ==="
cd "${DEPLOY_COMPOSE_DIR}"

# modification de la valeur de PROJECT_DIR dans le deploy/compose/.env
sed -i.bak "s|^PROJECT_DIR=.*|PROJECT_DIR=${ROOT_DIR}|" "${DEPLOY_COMPOSE_DIR}/.env"
rm -f "${DEPLOY_COMPOSE_DIR}/.env.bak"

docker compose up -d

echo
echo "✅ Tout est lancé."
echo "   - Bootstrap exécuté depuis: ${BOOTSTRAP_DIR}"
echo "   - Compose lancé depuis:     ${DEPLOY_COMPOSE_DIR}"
