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

# Configuration Streamlit
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
port = 8501\n\
address = 0.0.0.0\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = '#E63946'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#F1FAEE'\n\
textColor = '#1D3557'\n\
" > /root/.streamlit/config.toml

# Copie des fichiers de l'application
ARG APP_DIR=deploy/streamlit
COPY ${APP_DIR}/pages_old/ ./pages/

ARG DOCKER_DIR=docker/${PIXI_ENV}
COPY ${DOCKER_DIR}/assets/ ./assets/

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

ARG DOCKER_DIR=docker/${PIXI_ENV}
COPY ${DOCKER_DIR}/README_simplified.md .

ARG APP_DIR=deploy/streamlit
COPY ${APP_DIR}/app.py .

COPY --from=builder /root/.streamlit /root/.streamlit
COPY --from=builder /app/pages /app/pages
COPY --from=builder /app/assets /app/assets

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"

ENTRYPOINT []
CMD ["streamlit", "run", "app.py"]
