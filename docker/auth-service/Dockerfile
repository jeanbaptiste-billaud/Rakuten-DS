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


# ---------- Runtime ----------
FROM dhi.io/python:3.12-debian13 AS runtime

ARG PIXI_ENV

ENV PIXI_HOME=/opt/pixi
ENV PATH="${PIXI_HOME}/bin:/app/.pixi/envs/${PIXI_ENV}/bin:${PATH}"

WORKDIR /app

COPY --from=builder /opt/pixi /opt/pixi
COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /app/pixi.toml /app/pixi.toml
COPY --from=builder /app/pixi.lock /app/pixi.lock
COPY --from=infisical-cli /bin/infisical /usr/local/bin/infisical

# Code du service d'auth
COPY docker/${PIXI_ENV}/service.py app.py
COPY docker/${PIXI_ENV}/auth_utils.py .
COPY docker/${PIXI_ENV}/auth.json .

EXPOSE 8000

ENTRYPOINT []
# Lancement du service HTTP
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
