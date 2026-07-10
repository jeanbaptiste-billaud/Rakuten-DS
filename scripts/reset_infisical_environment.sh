#!/usr/bin/env bash
set -euo pipefail

source "$(git rev-parse --show-toplevel)/scripts/lib/project_paths.sh"

remove_container() {
  local name="$1"
  podman rm -f "$name" >/dev/null 2>&1 || true
}

remove_volume() {
  local name="$1"
  podman volume rm -f "$name" >/dev/null 2>&1 || true
}

remove_file() {
  local path="$1"
  rm -f "$path"
}

remove_container postgres
remove_container redis
remove_container infisical

remove_volume pgdata
remove_volume redis_data

remove_file "$PROJECT_ROOT/infisical-bootstrap.json"
remove_file "$PROJECT_ROOT/infisical-bootstrap.log"
remove_file "$TERRAFORM_DIR/infisical/terraform.auto.tfvars.json"
remove_file "$TERRAFORM_DIR/infisical/terraform-output.json"
remove_file "$TERRAFORM_DIR/infisical/terraform.tfstate"
remove_file "$TERRAFORM_DIR/infisical/runtime.env"

echo "Infisical environment reset."

cd "$PROJECT_ROOT"

ansible-playbook \
  -e "project_root=$PROJECT_ROOT" \
  infra/ansible/playbooks/bootstrap/bootstrap.yaml \
  --tags init

ansible-playbook \
  -e "project_root=$PROJECT_ROOT" \
  infra/ansible/playbooks/bootstrap/setup_infisical.yaml
