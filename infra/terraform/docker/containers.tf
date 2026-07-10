module "core" {
  source = "./modules/core"

  image_ids                = { for name, image in docker_image.images : name => image.image_id }
  volume_names             = merge(
    { for name, volume in docker_volume.volumes : name => volume.name },
    { logs_and_reports = docker_volume.volumes["logs"].name }
  )
  network_name             = docker_network.mlflow.name
  workdir                  = local.workdir
  mlflow_endpoint          = local.mlflow_endpoint
  mlflow_backend_store_uri = local.mlflow_backend_store_uri
  compose_dir              = local.compose_dir
  postgres_user            = local.stack_credentials.postgres_user
  postgres_password        = local.stack_credentials.postgres_password
  airflow_db               = var.airflow_db
  minio_root_user          = local.stack_credentials.minio_root_user
  minio_root_password      = local.stack_credentials.minio_root_password
  minio_port               = var.minio_port
  minio_mlflow_bucket      = var.minio_mlflow_bucket
  mlflow_s3_access_key     = local.stack_credentials.mlflow_s3_access_key
  mlflow_s3_secret_key     = local.stack_credentials.mlflow_s3_secret_key
  mlflow_host              = var.mlflow_host
  mlflow_port              = var.mlflow_port
  model_host               = var.model_host
  model_port               = var.model_port
}

module "monitoring" {
  source = "./modules/monitoring"

  image_ids      = { for name, image in docker_image.images : name => image.image_id }
  network_name   = docker_network.mlflow.name
  monitoring_dir = local.monitoring_dir

  depends_on = [module.core]
}

module "frontend" {
  source = "./modules/frontend"

  image_ids     = { for name, image in docker_image.images : name => image.image_id }
  volume_names  = merge(
    { for name, volume in docker_volume.volumes : name => volume.name },
    { logs_and_reports = docker_volume.volumes["logs"].name }
  )
  network_name  = docker_network.mlflow.name
  streamlit_dir = local.streamlit_dir

  depends_on = [
    module.core,
    module.monitoring,
    module.airflow
  ]
}

module "airflow" {
  source = "./modules/airflow"

  image_ids                 = { for name, image in docker_image.images : name => image.image_id }
  network_name              = docker_network.mlflow.name
  airflow_mounts            = local.airflow_mounts
  airflow_common_env        = local.airflow_common_env
  airflow_uid               = var.airflow_uid
  docker_gid                = var.docker_gid
  airflow_www_user_username = local.stack_credentials.airflow_www_user_username
  airflow_www_user_password = local.stack_credentials.airflow_www_user_password

  depends_on = [module.core, module.monitoring]
}
