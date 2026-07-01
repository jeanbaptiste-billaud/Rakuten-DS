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

RUN mkdir -p /src && touch /src/__init__.py
COPY src/data_module_df /src/data_module_df
COPY src/pipeline_steps_df/data /src
COPY src/utils /src/utils

RUN groupadd -g 1000 trainusr \
 && useradd -u 1000 -g trainusr -m -s /bin/bash trainusr

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

COPY --from=builder /src /src

ENV PYTHONPATH="/"

COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group
USER trainusr

ARG WORKDIR=/app
WORKDIR ${WORKDIR}

EXPOSE 8001
ENTRYPOINT []
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]
