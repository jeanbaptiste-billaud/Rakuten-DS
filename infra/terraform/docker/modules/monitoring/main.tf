terraform {
  required_version = ">= 1.5.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

resource "docker_container" "prometheus" {
  name  = "prometheus"
  image = var.image_ids["prometheus"]

  volumes {
    host_path      = var.monitoring_dir
    container_path = "/etc/prometheus"
  }

  ports {
    internal = 9090
    external = 9090
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "statsd_exporter" {
  name  = "statsd-exporter"
  image = var.image_ids["statsd_exporter"]

  command = [
    "--statsd.listen-udp=:8125",
    "--web.listen-address=:9102"
  ]

  ports {
    internal = 9102
    external = 9102
  }

  ports {
    internal = 8125
    external = 8125
    protocol = "udp"
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "grafana" {
  name  = "grafana"
  image = var.image_ids["grafana"]

  env = [
    "GF_AUTH_DISABLE_LOGIN_FORM=true",
    "GF_AUTH_ANONYMOUS_ENABLED=true",
    "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin"
  ]

  volumes {
    host_path      = "${var.monitoring_dir}/grafana/provisioning"
    container_path = "/etc/grafana/provisioning"
  }

  ports {
    internal = 3000
    external = 3000
  }

  networks_advanced {
    name = var.network_name
  }
}
