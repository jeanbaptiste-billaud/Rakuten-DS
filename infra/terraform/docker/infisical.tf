data "infisical_secrets" "docker_stack" {
  env_slug     = var.infisical_env
  workspace_id = var.infisical_project_id
  folder_path  = var.infisical_secret_path
}

data "infisical_secrets" "mlflow_service" {
  env_slug     = var.infisical_env
  workspace_id = var.infisical_project_id
  folder_path  = "/services/mlflow"
}
