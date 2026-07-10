variable "image_ids" {
  description = "Docker image IDs keyed by logical image name."
  type        = map(string)
}

variable "volume_names" {
  description = "Docker volume names keyed by logical volume name."
  type        = map(string)
}

variable "network_name" {
  description = "Docker network name used by the stack."
  type        = string
}

variable "workdir" {
  type = string
}

variable "mlflow_endpoint" {
  type = string
}

variable "mlflow_backend_store_uri" {
  type = string
}

variable "compose_dir" {
  type = string
}

variable "postgres_user" {
  type      = string
  sensitive = true
}

variable "postgres_password" {
  type      = string
  sensitive = true
}

variable "airflow_db" {
  type = string
}

variable "minio_root_user" {
  type      = string
  sensitive = true
}

variable "minio_root_password" {
  type      = string
  sensitive = true
}

variable "minio_port" {
  type = number
}

variable "minio_mlflow_bucket" {
  type = string
}

variable "mlflow_s3_access_key" {
  type      = string
  sensitive = true
}

variable "mlflow_s3_secret_key" {
  type      = string
  sensitive = true
}

variable "mlflow_host" {
  type = string
}

variable "mlflow_port" {
  type = number
}

variable "model_host" {
  type = string
}

variable "model_port" {
  type = number
}
