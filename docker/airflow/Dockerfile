ARG AIRFLOW_IMAGE_NAME="apache/airflow:3.1.3"
ARG INFISICAL_CLI_IMAGE=docker.io/infisical/cli:latest

FROM ${INFISICAL_CLI_IMAGE} AS infisical-cli

FROM ${AIRFLOW_IMAGE_NAME}

USER root
COPY --from=infisical-cli /bin/infisical /usr/local/bin/infisical
RUN apt-get update \
    && apt-get install -y docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

USER airflow
