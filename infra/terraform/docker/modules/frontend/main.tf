terraform {
  required_version = ">= 1.5.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

resource "docker_container" "api_gateway" {
  name    = "api-gateway"
  image   = var.image_ids["api_gateway"]
  restart = "unless-stopped"

  ports {
    internal = 8000
    external = 8000
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "auth_service" {
  name  = "auth-service"
  image = var.image_ids["auth_service"]

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "reverse_proxy" {
  name  = "reverse-proxy"
  image = var.image_ids["reverse_proxy"]

  ports {
    internal = 80
    external = 8085
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "streamlit_dashboard" {
  name    = "streamlit-dashboard"
  image   = var.image_ids["streamlit"]
  restart = "unless-stopped"

  env = ["PYTHONUNBUFFERED=1"]

  volumes {
    volume_name    = var.volume_names["logs_and_reports"]
    container_path = "/app/reports"
  }

  volumes {
    host_path      = "${var.streamlit_dir}/.streamlit"
    container_path = "/root/.streamlit"
  }

  ports {
    internal = 8501
    external = 8501
  }

  healthcheck {
    test     = ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
    interval = "30s"
    timeout  = "10s"
    retries  = 3
  }

  networks_advanced {
    name = var.network_name
  }
}
