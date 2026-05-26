# ---------- Builder ----------
FROM docker.io/jbbillaud/rakuten:pixi-latest AS builder

WORKDIR /app

ARG PIXI_CATALOG_DIR=docker/python_library_catalogs

COPY ${PIXI_CATALOG_DIR}/pixi.toml ./pixi.toml
COPY ${PIXI_CATALOG_DIR}/pixi.lock ./pixi.lock

RUN pixi install -e mlflow --locked \
 && pixi clean cache --yes


# ---------- Runtime ----------
FROM python:3.12-slim AS runtime

ENV PIXI_HOME=/opt/pixi
ENV PATH="${PIXI_HOME}/bin:/app/.pixi/envs/mlflow/bin:${PATH}"

WORKDIR /app

COPY --from=builder /opt/pixi /opt/pixi
COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /app/pixi.toml /app/pixi.toml
COPY --from=builder /app/pixi.lock /app/pixi.lock

EXPOSE 5000