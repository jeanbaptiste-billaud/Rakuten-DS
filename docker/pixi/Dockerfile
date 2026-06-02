FROM dhi.io/python:3.12-debian13-dev

ENV PIXI_HOME=/opt/pixi
ENV PATH="${PIXI_HOME}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ca-certificates \
    gzip \
    passwd \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | sh