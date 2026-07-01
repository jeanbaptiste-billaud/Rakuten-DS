ARG INFISICAL_CLI_IMAGE=docker.io/infisical/cli:latest

FROM ${INFISICAL_CLI_IMAGE} AS infisical-cli

FROM dhi.io/debian-base:trixie-debian13-dev

COPY --from=infisical-cli /bin/infisical /usr/local/bin/infisical

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget

RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc \
    && chmod +x mc \
    && ./mc --help

RUN apt-get purge -y --auto-remove wget \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["mc"]
