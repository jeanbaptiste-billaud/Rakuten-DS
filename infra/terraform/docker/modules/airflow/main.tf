terraform {
  required_version = ">= 1.5.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

resource "docker_container" "airflow_init" {
  name       = "airflow-init"
  image      = var.image_ids["airflow"]
  entrypoint = ["/bin/bash"]
  must_run   = false
  command = [
    "-c",
    <<-EOT
      if [[ -z "$${AIRFLOW_UID}" ]]; then
        echo
        echo -e "\033[1;33mWARNING!!!: AIRFLOW_UID not set!\e[0m"
        echo "If you are on Linux, you SHOULD follow the instructions below to set "
        echo "AIRFLOW_UID environment variable, otherwise files will be owned by root."
        echo
        export AIRFLOW_UID=$(id -u)
      fi
      one_meg=1048576
      mem_available=$(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE) / one_meg))
      cpus_available=$(grep -cE 'cpu[0-9]+' /proc/stat)
      disk_available=$(df / | tail -1 | awk '{print $4}')
      warning_resources="false"
      if (( mem_available < 4000 )) ; then
        echo
        echo -e "\033[1;33mWARNING!!!: Not enough memory available for Docker.\e[0m"
        echo "At least 4GB of memory required. You have $(numfmt --to iec $((mem_available * one_meg)))"
        warning_resources="true"
      fi
      if (( cpus_available < 2 )); then
        echo
        echo -e "\033[1;33mWARNING!!!: Not enough CPUS available for Docker.\e[0m"
        echo "At least 2 CPUs recommended. You have $${cpus_available}"
        warning_resources="true"
      fi
      if (( disk_available < one_meg * 10 )); then
        echo
        echo -e "\033[1;33mWARNING!!!: Not enough Disk space available for Docker.\e[0m"
        echo "At least 10 GBs recommended. You have $(numfmt --to iec $((disk_available * 1024 )))"
        warning_resources="true"
      fi
      if [[ $${warning_resources} == "true" ]]; then
        echo
        echo -e "\033[1;33mWARNING!!!: You have not enough resources to run Airflow (see above)!\e[0m"
      fi
      echo "Creating missing opt dirs if missing:"
      mkdir -v -p /opt/airflow/{logs,dags,plugins,config}
      echo "Airflow version:"
      /entrypoint airflow version
      echo "Running airflow config list to create default config file if missing."
      /entrypoint airflow config list >/dev/null
      echo "Change ownership of files in /opt/airflow to $${AIRFLOW_UID}:0"
      chown -R "$${AIRFLOW_UID}:0" /opt/airflow/
      echo "Change ownership of files in shared volumes to $${AIRFLOW_UID}:0"
      chown -v -R "$${AIRFLOW_UID}:0" /opt/airflow/{logs,dags,plugins,config}
    EOT
  ]

  env = concat(var.airflow_common_env, [
    "_AIRFLOW_DB_MIGRATE=true",
    "_AIRFLOW_WWW_USER_CREATE=true",
    "_AIRFLOW_WWW_USER_USERNAME=${var.airflow_www_user_username}",
    "_AIRFLOW_WWW_USER_PASSWORD=${var.airflow_www_user_password}"
  ])

  user = "0:0"

  dynamic "volumes" {
    for_each = var.airflow_mounts

    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "airflow_dag_processor" {
  name    = "airflow-dag-processor"
  image   = var.image_ids["airflow"]
  command = ["dag-processor"]
  restart = "always"
  user    = "${var.airflow_uid}:0"

  env       = var.airflow_common_env
  group_add = [var.docker_gid]

  dynamic "volumes" {
    for_each = var.airflow_mounts

    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  healthcheck {
    test         = ["CMD-SHELL", "airflow jobs check --job-type DagProcessorJob --hostname \"$${HOSTNAME}\""]
    interval     = "30s"
    timeout      = "10s"
    retries      = 5
    start_period = "30s"
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "airflow_apiserver" {
  name    = "airflow-apiserver"
  image   = var.image_ids["airflow"]
  command = ["api-server"]
  restart = "always"
  user    = "${var.airflow_uid}:0"

  env       = var.airflow_common_env
  group_add = [var.docker_gid]

  dynamic "volumes" {
    for_each = var.airflow_mounts

    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  ports {
    internal = 8080
    external = 8080
  }

  healthcheck {
    test         = ["CMD", "curl", "--fail", "http://localhost:8080/api/v2/version"]
    interval     = "30s"
    timeout      = "10s"
    retries      = 5
    start_period = "30s"
  }

  networks_advanced {
    name = var.network_name
  }
}

resource "docker_container" "airflow_scheduler" {
  name    = "airflow-scheduler"
  image   = var.image_ids["airflow"]
  command = ["scheduler"]
  restart = "always"
  user    = "${var.airflow_uid}:0"

  env       = var.airflow_common_env
  group_add = [var.docker_gid]

  dynamic "volumes" {
    for_each = var.airflow_mounts

    content {
      host_path      = volumes.value.host_path
      container_path = volumes.value.container_path
    }
  }

  healthcheck {
    test         = ["CMD", "curl", "--fail", "http://localhost:8974/health"]
    interval     = "30s"
    timeout      = "10s"
    retries      = 5
    start_period = "30s"
  }

  networks_advanced {
    name = var.network_name
  }
}
