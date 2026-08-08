FROM docker.io/jbbillaud/rakuten:pixi-latest AS builder
WORKDIR /app
ARG PIXI_CATALOG_DIR
ARG PIXI_ENV
COPY ${PIXI_CATALOG_DIR}/pixi.toml ./pixi.toml
COPY ${PIXI_CATALOG_DIR}/pixi.lock ./pixi.lock
RUN pixi install -e ${PIXI_ENV} --locked && pixi clean cache --yes
RUN mkdir -p /src && touch /src/__init__.py
COPY src/security /src/security

FROM dhi.io/python:3.12-debian13-dev AS runtime
ARG PIXI_ENV
ENV PIXI_HOME=/opt/pixi
ENV PATH="${PIXI_HOME}/bin:/app/.pixi/envs/${PIXI_ENV}/bin:${PATH}"
ENV PYTHONPATH="/"
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-client \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/pixi /opt/pixi
COPY --from=builder /app/.pixi /app/.pixi
COPY --from=builder /app/pixi.toml /app/pixi.toml
COPY --from=builder /app/pixi.lock /app/pixi.lock
COPY --from=builder /src /src
RUN groupadd -g 1000 backupusr \
 && useradd -u 1000 -g backupusr -m -s /bin/bash backupusr
USER backupusr
WORKDIR /app
ENTRYPOINT []
