variable "docker_host" {
  description = "Docker-compatible API endpoint. For rootless Podman: unix:///run/user/<uid>/podman/podman.sock."
  type        = string
  default     = "unix:///run/user/1000/podman/podman.sock"
}

variable "project_dir" {
  description = "Absolute path to deploy_branch. Defaults to this repository's deploy_branch directory."
  type        = string
  default     = null
}

variable "airflow_uid" {
  description = "UID used by Airflow containers for mounted file ownership."
  type        = string
  default     = "1000"
}

variable "docker_gid" {
  description = "GID allowed to access the container engine socket from Airflow containers."
  type        = string
  default     = "1001"
}

variable "container_engine_socket_path" {
  description = "Host path mounted as /var/run/docker.sock for Airflow DockerOperator tasks."
  type        = string
  default     = "/var/run/docker.sock"
}

variable "airflow_db" {
  description = "Airflow database name."
  type        = string
  default     = "airflow_db"
}

variable "mlflow_db" {
  description = "MLflow database name."
  type        = string
  default     = "mlflow_db"
}

variable "minio_version" {
  description = "MinIO image tag."
  type        = string
  default     = "RELEASE.2025-09-07T16-13-09Z"
}

variable "minio_host" {
  description = "Internal MinIO hostname."
  type        = string
  default     = "minio"
}

variable "minio_port" {
  description = "Internal MinIO API port."
  type        = number
  default     = 9000
}

variable "minio_mlflow_bucket" {
  description = "Bucket used by MLflow artifacts."
  type        = string
  default     = "mlflow-artifacts"
}

variable "mlflow_version" {
  description = "MLflow image tag suffix."
  type        = string
  default     = "v3.8.1"
}

variable "mlflow_host" {
  description = "MLflow bind host."
  type        = string
  default     = "0.0.0.0"
}

variable "mlflow_port" {
  description = "MLflow bind port."
  type        = number
  default     = 5000
}

variable "model_host" {
  description = "BentoML model-serving bind host."
  type        = string
  default     = "0.0.0.0"
}

variable "model_port" {
  description = "BentoML model-serving port."
  type        = number
  default     = 8002
}

variable "infisical_domain" {
  description = "Infisical URL used by the provider and injected into Airflow for workload secret retrieval."
  type        = string
  default     = "http://infisical:8080"
}

variable "infisical_provider_client_id" {
  description = "Universal Auth client ID used by Terraform to read Infisical secrets for the Docker stack."
  type        = string
  sensitive   = true
}

variable "infisical_provider_client_secret" {
  description = "Universal Auth client secret used by Terraform to read Infisical secrets for the Docker stack."
  type        = string
  sensitive   = true
}

variable "infisical_secret_path" {
  description = "Folder path inside Infisical containing the Docker stack credentials."
  type        = string
  default     = "/"
}

variable "infisical_env" {
  description = "Infisical environment slug used by the provider and injected into Airflow."
  type        = string
  default     = "dev"
}

variable "infisical_project_id" {
  description = "Infisical project ID used by the provider and injected into Airflow. Required for the stack to read secrets."
  type        = string
  default     = ""
}

variable "infisical_identity_ids" {
  description = "Infisical machine identity UUIDs keyed by Terraform identity name, such as data-ingestion-id."
  type        = map(string)
  default     = {}
}
