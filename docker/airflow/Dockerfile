ARG AIRFLOW_IMAGE_NAME="apache/airflow:3.1.3"

FROM ${AIRFLOW_IMAGE_NAME}

USER root
RUN apt-get update \
    && apt-get install -y docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

USER airflow