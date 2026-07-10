variable "image_ids" {
  description = "Docker image IDs keyed by logical image name."
  type        = map(string)
}

variable "network_name" {
  description = "Docker network name used by the stack."
  type        = string
}

variable "airflow_mounts" {
  type = list(object({
    host_path      = string
    container_path = string
  }))
}

variable "airflow_common_env" {
  type      = list(string)
  sensitive = true
}

variable "airflow_uid" {
  type = string
}

variable "docker_gid" {
  type = string
}

variable "airflow_www_user_username" {
  type = string
}

variable "airflow_www_user_password" {
  type      = string
  sensitive = true
}
