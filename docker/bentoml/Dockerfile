FROM docker:29.1.1-cli

WORKDIR /app

# Installer python et les dépendances
RUN apk add --no-cache\
    python3 \
    py3-pip \
    python3-dev \
    build-base \
    openblas-dev \
    linux-headers \
    pkgconfig\
    && ln -sf python3 /usr/bin/python

# créer env virtuel
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Installer les dépendances
COPY docker/bentoml/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download fr_core_news_sm

# Copie des utilitaires
COPY src/data_module_df /app/src/data_module_df
COPY src/utils /app/src/utils
COPY src/model_serving /app

ENV PYTHONPATH="/app"
