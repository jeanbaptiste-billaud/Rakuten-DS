resource "docker_network" "mlflow" {
  name = local.docker_config.network.name
}

resource "docker_volume" "volumes" {
  for_each = local.volumes
  name     = each.value
}
