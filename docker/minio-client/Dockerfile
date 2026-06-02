FROM dhi.io/debian-base:trixie-debian13-dev

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