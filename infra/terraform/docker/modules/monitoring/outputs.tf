output "container_names" {
  value = {
    prometheus      = docker_container.prometheus.name
    statsd_exporter = docker_container.statsd_exporter.name
    grafana         = docker_container.grafana.name
  }
}
