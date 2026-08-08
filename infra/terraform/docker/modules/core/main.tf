terraform {
  required_version = ">= 1.5.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

resource "docker_container" "postgres" {
  name    = "postgres"
  image   = var.image_ids["postgres"]
  restart = "always"

  env = [
    "POSTGRES_USER=${var.postgres_user}",
    "POSTGRES_PASSWORD=${var.postgres_password}",
    "POSTGRES_DB=${var.airflow_db}"
  ]

  volumes {
    volume_name    = var.volume_names["pgdata"]
    container_path = "/var/lib/postgresql/17/data"
  }

  volumes {
    host_path      = "${var.compose_dir}/init_db.sql"
    container_path = "/docker-entrypoint-initdb.d/init_db.sql"
    read_only      = true
  }

  healthcheck {
    test         = ["CMD-SHELL", "pg_isready -h 127.0.0.1 -U \"$${POSTGRES_USER}\" -d \"$${POSTGRES_DB}\""]
    interval     = "10s"
    retries      = 5
    start_period = "5s"
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "minio" {
  name  = "minio-S3-storage"
  image = var.image_ids["minio"]

  env = [
    "MINIO_ROOT_USER=${var.minio_root_user}",
    "MINIO_ROOT_PASSWORD=${var.minio_root_password}"
  ]

  command = ["server", "/data", "--console-address", ":9001"]

  volumes {
    volume_name    = var.volume_names["minio_data"]
    container_path = "/data"
  }

  ports {
    internal = 9000
    external = 9000
  }

  ports {
    internal = 9001
    external = 9001
  }

  healthcheck {
    test     = ["CMD", "curl", "-f", "http://localhost:${var.minio_port}/minio/health/live"]
    interval = "5s"
    timeout  = "3s"
    retries  = 20
  }

  networks_advanced {
    name    = var.network_name
    aliases = ["minio"]
  }
}

resource "docker_container" "mlflow_server" {
  name  = "mlflow-server"
  image = var.image_ids["mlflow_server"]

  env = [
    "MLFLOW_BACKEND_STORE_URI=${var.mlflow_backend_store_uri}",
    "MLFLOW_ARTIFACT_URI=http://mlflow-artifacts:5000",
    "AWS_ACCESS_KEY_ID=${var.mlflow_s3_access_key}",
    "AWS_SECRET_ACCESS_KEY=${var.mlflow_s3_secret_key}",
    "MLFLOW_S3_ENDPOINT_URL=${var.mlflow_endpoint}",
    "MLFLOW_HOST=${var.mlflow_host}",
    "MLFLOW_PORT=${var.mlflow_port}"
  ]

  command = [
    "mlflow", "server",
    "--backend-store-uri", var.mlflow_backend_store_uri,
    "--host", var.mlflow_host,
    "--port", tostring(var.mlflow_port),
    "--default-artifact-root", "s3://${var.minio_mlflow_bucket}/",
    "--allowed-hosts", "localhost:5000, *"
  ]

  ports {
    internal = 5000
    external = 5000
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "drift_detector" {
  name    = "drift-detector"
  image   = var.image_ids["drift_detector"]
  restart = "unless-stopped"

  env = [
    "MODEL_SERVING_URL=http://model-serving:8002",
    "WORKDIR=${var.workdir}"
  ]

  volumes {
    volume_name    = var.volume_names["logs_and_reports"]
    container_path = "/app/evidently"
  }

  ports {
    internal = 8003
    external = 8003
  }

  healthcheck {
    test     = ["CMD", "curl", "-f", "http://localhost:8003/health"]
    interval = "30s"
    timeout  = "10s"
    retries  = 3
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "model_serving" {
  name    = "model-serving"
  image   = var.image_ids["model_serving"]
  restart = "unless-stopped"

  env = [
    "BENTOML_PORT=${var.model_port}",
    "BENTOML_HOST=${var.model_host}"
  ]

  volumes {
    volume_name    = var.volume_names["logs_and_reports"]
    container_path = "/logs_and_reports"
  }

  ports {
    internal = var.model_port
    external = var.model_port
  }

  healthcheck {
    test     = ["CMD-SHELL", "curl -f http://localhost:${var.model_port}/healthz || exit 1"]
    interval = "10s"
    timeout  = "3s"
    retries  = 10
  }

  networks_advanced {
    name = var.network_name
  }
}
