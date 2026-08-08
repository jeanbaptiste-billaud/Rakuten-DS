locals {
  docker_config = yamldecode(file("${path.module}/../../config/docker.yaml"))

  project_dir     = var.project_dir == null ? abspath("${path.module}/../../../deploy_branch") : var.project_dir
  deploy_dir      = "${local.project_dir}/deploy"
  compose_dir     = "${local.deploy_dir}/compose"
  airflow_dir     = "${local.deploy_dir}/airflow"
  monitoring_dir  = "${local.deploy_dir}/monitoring"
  streamlit_dir   = "${local.deploy_dir}/streamlit"
  workdir         = "/app"
  mlflow_endpoint = "http://${var.minio_host}:${var.minio_port}"

  images = {
    postgres        = "dhi.io/postgres:17"
    minio           = "minio/minio:${var.minio_version}"
    mlflow_server   = "jbbillaud/rakuten:mlflow-${var.mlflow_version}"
    drift_detector  = "jbbillaud/rakuten:evidently-v0.7.20"
    prometheus      = "prom/prometheus:latest"
    statsd_exporter = "prom/statsd-exporter:latest"
    grafana         = "grafana/grafana:latest"
    model_serving   = "jbbillaud/rakuten:text-model-latest"
    api_gateway     = "jbbillaud/rakuten:fastapi-v0.128.0"
    auth_service    = "jbbillaud/rakuten:authentification"
    reverse_proxy   = "jbbillaud/rakuten:nginx-v3.23"
    streamlit       = "jbbillaud/rakuten:streamlit-v1.29.0"
    airflow         = "dhi.io/airflow:3-compat"
  }

  infisical_stack_secrets  = data.infisical_secrets.docker_stack.secrets
  infisical_mlflow_secrets = data.infisical_secrets.mlflow_service.secrets

  stack_credentials = {
    postgres_user             = nonsensitive(local.infisical_stack_secrets["POSTGRES_AIRFLOW_USER"].value)
    postgres_password         = nonsensitive(local.infisical_stack_secrets["POSTGRES_AIRFLOW_PASSWORD"].value)
    minio_root_user           = nonsensitive(local.infisical_stack_secrets["MINIO_ROOT_USER"].value)
    minio_root_password       = nonsensitive(local.infisical_stack_secrets["MINIO_ROOT_PASSWORD"].value)
    mlflow_s3_access_key      = nonsensitive(local.infisical_mlflow_secrets["MLFLOW_S3_ACCESS_KEY"].value)
    mlflow_s3_secret_key      = nonsensitive(local.infisical_mlflow_secrets["MLFLOW_S3_SECRET_KEY"].value)
    airflow_www_user_username = try(nonsensitive(local.infisical_stack_secrets["AIRFLOW_WWW_USER_USERNAME"].value), "airflow")
    airflow_www_user_password = try(nonsensitive(local.infisical_stack_secrets["AIRFLOW_WWW_USER_PASSWORD"].value), "airflow")
  }

  mlflow_backend_store_uri = "postgresql://${local.stack_credentials.postgres_user}:${local.stack_credentials.postgres_password}@postgres/${var.mlflow_db}"

  volumes = local.docker_config.volumes

  airflow_mounts = [
    {
      host_path      = "${local.airflow_dir}/dags"
      container_path = "/opt/airflow/dags"
    },
    {
      host_path      = "${local.airflow_dir}/config"
      container_path = "/opt/airflow/config"
    },
    {
      host_path      = "${local.airflow_dir}/plugins"
      container_path = "/opt/airflow/plugins"
    },
    {
      host_path      = "${local.airflow_dir}/logs"
      container_path = "/opt/airflow/logs"
    },
    {
      host_path      = local.compose_dir
      container_path = "/opt/airflow/compose"
    },
    {
      host_path      = var.container_engine_socket_path
      container_path = "/var/run/docker.sock"
    }
  ]

  infisical_identity_env = flatten([
    for identity, identity_id in var.infisical_identity_ids :
    "INFISICAL_${replace(upper(identity), "-", "_")}_IDENTITY_ID=${identity_id}"
  ])

  airflow_common_env = concat([
    "AIRFLOW__CORE__EXECUTOR=LocalExecutor",
    "AIRFLOW__CORE__AUTH_MANAGER=airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager",
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://${local.stack_credentials.postgres_user}:${local.stack_credentials.postgres_password}@postgres/${var.airflow_db}",
    "AIRFLOW_CONN_POSTGRES_DEFAULT=postgresql://${local.stack_credentials.postgres_user}:${local.stack_credentials.postgres_password}@postgres:5432/postgres",
    "AIRFLOW__CORE__FERNET_KEY=",
    "AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true",
    "AIRFLOW__CORE__LOAD_EXAMPLES=false",
    "AIRFLOW__CORE__EXECUTION_API_SERVER_URL=http://airflow-apiserver:8080/execution/",
    "AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK=true",
    "_PIP_ADDITIONAL_REQUIREMENTS=",
    "AIRFLOW_CONFIG=/opt/airflow/config/airflow.cfg",
    "WORKDIR=${local.workdir}",
    "AIRFLOW_UID=${var.airflow_uid}",
    "DOCKER_GID=${var.docker_gid}",
    "DRIFT_DETECTOR_URL=http://drift-detector:8003",
    "INFISICAL_DOMAIN=${var.infisical_domain}",
    "INFISICAL_ENV=${var.infisical_env}",
    "INFISICAL_PROJECT_ID=${var.infisical_project_id}",
    "CONTAINER_NETWORK=${local.docker_config.network.name}",
    "SPIRE_SOCKET_VOLUME=${local.docker_config.volumes.spire_socket}",
    "SPIFFE_ENDPOINT_SOCKET=unix:///run/spire/sockets/agent.sock",
    "AIRFLOW__METRICS__STATSD_ON=True",
    "AIRFLOW__METRICS__STATSD_HOST=statsd-exporter",
    "AIRFLOW__METRICS__STATSD_PORT=8125",
    "AIRFLOW__METRICS__STATSD_PREFIX=airflow"
  ], local.infisical_identity_env)
}
