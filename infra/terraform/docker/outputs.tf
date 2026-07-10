output "network_name" {
  value = docker_network.mlflow.name
}

output "container_names" {
  value = merge(
    module.core.container_names,
    module.monitoring.container_names,
    module.frontend.container_names,
    module.airflow.container_names
  )
}
