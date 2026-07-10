output "container_names" {
  value = {
    airflow_init          = docker_container.airflow_init.name
    airflow_dag_processor = docker_container.airflow_dag_processor.name
    airflow_apiserver     = docker_container.airflow_apiserver.name
    airflow_scheduler     = docker_container.airflow_scheduler.name
  }
}
