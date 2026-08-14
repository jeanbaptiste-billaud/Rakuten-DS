output "container_names" {
  value = {
    postgres       = docker_container.postgres.name
    minio          = docker_container.minio.name
    mlflow_server  = docker_container.mlflow_server.name
    drift_detector = docker_container.drift_detector.name
    model_serving  = docker_container.model_serving.name
  }
}
