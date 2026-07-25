#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
ANSIBLE_PLAYBOOK_DIR="${PROJECT_ROOT}/infra/ansible/playbooks"
TERRAFORM_DIR="${PROJECT_ROOT}/infra/terraform/docker"
INFISICAL_BOOTSTRAP_STATE_FILE="${PROJECT_ROOT}/infisical-bootstrap.json"
INFISICAL_BOOTSTRAP_STATE_FILE_FALLBACK="${HOME}/.config/rakuten/infisical/bootstrap.json"
INFISICAL_RUNTIME_ENV_FILE="${PROJECT_ROOT}/runtime.env"
INFISICAL_RUNTIME_ENV_FILE_FALLBACK="${HOME}/.config/rakuten/infisical/runtime.env"
INFISICAL_OUTPUTS_FILE="${PROJECT_ROOT}/terraform-output.json"
INFISICAL_OUTPUTS_FILE_FALLBACK="${HOME}/.config/rakuten/infisical/terraform-output.json"
PODMAN_SOCKET_PATH="unix:///run/user/$(id -u)/podman/podman.sock"
ANSIBLE_LOCAL_TEMP="${ANSIBLE_LOCAL_TEMP:-/tmp/ansible-local}"
ANSIBLE_REMOTE_TEMP="${ANSIBLE_REMOTE_TEMP:-/tmp/ansible-remote}"

log() {
  printf '[deploy] %s\n' "$*"
}

load_infisical_runtime_env() {
  local runtime_env_file=""

  if [[ -f "${INFISICAL_RUNTIME_ENV_FILE}" ]]; then
    runtime_env_file="${INFISICAL_RUNTIME_ENV_FILE}"
  elif [[ -f "${INFISICAL_RUNTIME_ENV_FILE_FALLBACK}" ]]; then
    runtime_env_file="${INFISICAL_RUNTIME_ENV_FILE_FALLBACK}"
  else
    return 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${runtime_env_file}"
  set +a
}

load_infisical_terraform_outputs() {
  local outputs_file=""

  if [[ -f "${INFISICAL_OUTPUTS_FILE}" ]]; then
    outputs_file="${INFISICAL_OUTPUTS_FILE}"
  elif [[ -f "${INFISICAL_OUTPUTS_FILE_FALLBACK}" ]]; then
    outputs_file="${INFISICAL_OUTPUTS_FILE_FALLBACK}"
  else
    return 1
  fi

  command -v jq >/dev/null 2>&1 || return 1

  export INFISICAL_PROJECT_ID
  export INFISICAL_ENV
  export INFISICAL_DOMAIN
  export INFISICAL_DEPLOY_ID_CLIENT_ID
  export INFISICAL_DEPLOY_ID_CLIENT_SECRET

  INFISICAL_PROJECT_ID="$(jq -r '.project_id.value // empty' "${outputs_file}")"
  INFISICAL_DEPLOY_ID_CLIENT_ID="$(jq -r '.terraform_identity_universal_auth.value.client_id // empty' "${outputs_file}")"
  INFISICAL_DEPLOY_ID_CLIENT_SECRET="$(jq -r '.terraform_identity_universal_auth.value.client_secret // empty' "${outputs_file}")"
  INFISICAL_ENV="${INFISICAL_ENV:-dev}"
  INFISICAL_DOMAIN="${INFISICAL_DOMAIN:-http://127.0.0.1:8080}"

  [[ -n "${INFISICAL_PROJECT_ID}" && -n "${INFISICAL_DEPLOY_ID_CLIENT_ID}" && -n "${INFISICAL_DEPLOY_ID_CLIENT_SECRET}" ]]
}

prepare_infisical_provider_env() {
  export TF_VAR_docker_host="${TF_VAR_docker_host:-${PODMAN_SOCKET_PATH}}"
  if load_infisical_runtime_env || load_infisical_terraform_outputs; then
    export TF_VAR_infisical_project_id="${INFISICAL_PROJECT_ID:?INFISICAL_PROJECT_ID is missing from Infisical state}"
    export TF_VAR_infisical_env="${INFISICAL_ENV:-dev}"
    export TF_VAR_infisical_domain="${TF_VAR_infisical_domain:-http://127.0.0.1:8080}"
    export TF_VAR_infisical_secret_path="/"

    if [[ -n "${INFISICAL_DEPLOY_ID_CLIENT_ID:-}" && -n "${INFISICAL_DEPLOY_ID_CLIENT_SECRET:-}" ]]; then
      export TF_VAR_infisical_provider_client_id="${INFISICAL_DEPLOY_ID_CLIENT_ID}"
      export TF_VAR_infisical_provider_client_secret="${INFISICAL_DEPLOY_ID_CLIENT_SECRET}"
      return 0
    fi
  fi

  return 1
}

run_recovery() {
  log "Recovering Infisical stack"
  ansible-playbook \
    "${ANSIBLE_PLAYBOOK_DIR}/deploy/recover_infisical.yaml" \
    --limit deploy
}

run_bootstrap_infisical() {
  log "Bootstrapping Infisical state"
  ansible-playbook \
    "${ANSIBLE_PLAYBOOK_DIR}/bootstrap/setup_infisical.yaml" \
    --limit deploy \
    -e "infisical_bootstrap_state_file=${INFISICAL_BOOTSTRAP_STATE_FILE}"
}

run_terraform() {
  log "Applying Terraform Docker stack"
  export TF_IN_AUTOMATION=1
  export TF_INPUT=0

  terraform -chdir="${TERRAFORM_DIR}" init -input=false
  terraform -chdir="${TERRAFORM_DIR}" apply -auto-approve -input=false
}

cd "${PROJECT_ROOT}"

mkdir -p "${ANSIBLE_LOCAL_TEMP}" "${ANSIBLE_REMOTE_TEMP}"
export ANSIBLE_LOCAL_TEMP ANSIBLE_REMOTE_TEMP

if ! prepare_infisical_provider_env; then
  log "Infisical state missing or incomplete, running bootstrap first"
  run_bootstrap_infisical
  if ! prepare_infisical_provider_env; then
    log "Infisical state is still incomplete after bootstrap. Expected ${INFISICAL_RUNTIME_ENV} or ${INFISICAL_OUTPUTS_FILE}."
    exit 1
  fi
fi

run_terraform
