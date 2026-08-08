FROM docker.io/jbbillaud/rakuten:pixi-latest AS builder
WORKDIR /app
ARG PIXI_CATALOG_DIR
ARG PIXI_ENV
COPY ${PIXI_CATALOG_DIR}/pixi.toml ./pixi.toml
COPY ${PIXI_CATALOG_DIR}/pixi.lock ./pixi.lock
RUN pixi install -e ${PIXI_ENV} --locked && pixi clean cache --yes
RUN mkdir -p /src && touch /src/__init__.py
COPY src/security /src/security
COPY src/minio /src/minio

FROM dhi.io/python:3.12-debian13 AS runtime
ARG PIXI_ENV
ENV PIXI_HOME=/opt/pixi
ENV PATH="${PIXI_HOME}/bin:/app/.pixi/envs/${PIXI_ENV}/bin:${PATH}"
ENV PYTHONPATH="/"
COPY --from=builder /opt/pixi /opt/pixi
COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /app/pixi.toml /app/pixi.toml
COPY --from=builder /app/pixi.lock /app/pixi.lock
COPY --from=builder /src /src
ENTRYPOINT []

