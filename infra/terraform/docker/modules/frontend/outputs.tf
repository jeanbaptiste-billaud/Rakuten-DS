output "container_names" {
  value = {
    api_gateway         = docker_container.api_gateway.name
    auth_service        = docker_container.auth_service.name
    reverse_proxy       = docker_container.reverse_proxy.name
    streamlit_dashboard = docker_container.streamlit_dashboard.name
  }
}
