#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Erreur : impossible de déterminer la racine du projet Git." >&2
    exit 1
}

export PROJECT_ROOT

export ANSIBLE_DIR="${PROJECT_ROOT}/infra/ansible"
export TERRAFORM_DIR="${PROJECT_ROOT}/infra/terraform"
export INFISICAL_BOOTSTRAP_FILE="${PROJECT_ROOT}/infisical-bootstrap.json"
export INFISICAL_BOOTSTRAP_LOG="${PROJECT_ROOT}/infisical-bootstrap.log"
