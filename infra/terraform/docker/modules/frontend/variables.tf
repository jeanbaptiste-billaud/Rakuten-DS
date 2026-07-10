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

variable "streamlit_dir" {
  type = string
}
