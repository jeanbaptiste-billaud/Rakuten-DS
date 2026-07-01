# ---------- Builder ----------
ARG INFISICAL_CLI_IMAGE=docker.io/infisical/cli:latest

FROM ${INFISICAL_CLI_IMAGE} AS infisical-cli

FROM docker.io/jbbillaud/rakuten:pixi-latest AS builder

WORKDIR /app

ARG PIXI_CATALOG_DIR
ARG PIXI_ENV

COPY ${PIXI_CATALOG_DIR}/pixi.toml ./pixi.toml
COPY ${PIXI_CATALOG_DIR}/pixi.lock ./pixi.lock

RUN pixi install -e ${PIXI_ENV} --locked \
 && pixi clean cache --yes

RUN groupadd -g 1000 trainusr \
 && useradd -u 1000 -g trainusr -m -s /usr/sbin/nologin trainusr

COPY src/model_serving /model_serving
COPY src/data_module_df /model_serving/src/data_module_df
COPY src/models_module_df /model_serving/src/models_module_df
COPY src/utils /model_serving/src/utils
RUN chmod +x /model_serving/model_building.sh

RUN chown -R trainusr:trainusr /model_serving

FROM dhi.io/docker:29-cli AS runtime

WORKDIR /app

COPY --from=builder /opt/pixi /opt/pixi
COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /app/pixi.toml /app/pixi.toml
COPY --from=builder /app/pixi.lock /app/pixi.lock
COPY --from=infisical-cli /bin/infisical /usr/local/bin/infisical

ARG PIXI_ENV

ENV PIXI_HOME=/opt/pixi
ENV PATH="${PIXI_HOME}/bin:/app/.pixi/envs/${PIXI_ENV}/bin:${PATH}"

# Copie des utilitaires
COPY --from=builder /model_serving /model_serving

# Copie des utilisateurs
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

USER trainusr

ARG WORKDIR=/app
WORKDIR ${WORKDIR}

ENV PYTHONPATH="/"

ENTRYPOINT []
