terraform {
  required_version = ">= 1.5.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
    infisical = {
      source  = "infisical/infisical"
      version = "~> 0.19.1"
    }
  }
}

provider "docker" {
  host = var.docker_host
}

provider "infisical" {
  host = var.infisical_domain
  auth = {
    universal = {
      client_id     = var.infisical_provider_client_id
      client_secret = var.infisical_provider_client_secret
    }
  }
}
