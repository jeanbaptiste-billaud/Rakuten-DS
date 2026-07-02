variable "infisical_host" {
  type        = string
  description = "Infisical API host, for example http://localhost:8080."
}

variable "bootstrap_token" {
  type        = string
  description = "Bootstrap machine identity token used by Terraform to configure Infisical."
  sensitive   = true
}

variable "organization_id" {
  type        = string
  description = "Infisical organization ID."
}

variable "project_name" {
  type        = string
  description = "Infisical project name."
}

variable "project_slug" {
  type        = string
  description = "Infisical project slug."
}

variable "environment_slug" {
  type        = string
  description = "Environment slug managed by the bootstrap."
  default     = "dev"
}

variable "identities" {
  type = map(object({
    path    = string
    actions = optional(list(string), ["describeSecret", "readValue"])
  }))
  description = "Machine identities and their scoped secret path."
}

