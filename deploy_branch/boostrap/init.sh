#!/usr/bin/env bash
set -euo pipefail

source "$(git rev-parse --show-toplevel)/scripts/lib/project_paths.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
ANSIBLE_CONFIG_FILE="${PROJECT_ROOT}/ansible.cfg"

export PROJECT_ROOT
export ANSIBLE_CONFIG="${ANSIBLE_CONFIG_FILE}"
export ANSIBLE_LOCAL_TEMP="${ANSIBLE_LOCAL_TEMP:-/tmp/ansible-local}"
export ANSIBLE_REMOTE_TEMP="${ANSIBLE_REMOTE_TEMP:-/tmp/ansible-remote}"

mkdir -p "${ANSIBLE_LOCAL_TEMP}" "${ANSIBLE_REMOTE_TEMP}"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  source "${SCRIPT_DIR}/.env"
  set +a
fi

ensure_ansible() {
  if command -v ansible-playbook >/dev/null 2>&1 && command -v ansible-galaxy >/dev/null 2>&1; then
    return
  fi

  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install ansible-core." >&2
    exit 1
  fi

  sudo apt-get update
  sudo apt-get install -y ansible-core
}

install_collections() {
  ansible-galaxy collection install -r "${PROJECT_ROOT}/infra/ansible/requirements.yaml"
}

run_playbook() {
  local playbook="$1"
  shift
  ansible-playbook "$playbook" "$@"
}

ensure_ansible

cd "${PROJECT_ROOT}"

install_collections

run_playbook \
  infra/ansible/playbooks/common/install_local_softwares.yaml \
  --limit deploy \
  -e podman_dhi_login_required=true

run_playbook \
  infra/ansible/playbooks/bootstrap/boostrap_pre_vault.yaml \
  --limit deploy

run_playbook \
  infra/ansible/playbooks/bootstrap/bootstrap.yaml \
  --limit deploy

run_playbook \
  infra/ansible/playbooks/bootstrap/setup_infisical.yaml \
  --limit deploy
