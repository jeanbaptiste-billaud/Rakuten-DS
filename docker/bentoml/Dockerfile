FROM docker:29.1.1-cli-alpine3.22

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

ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# installe dans le venv
COPY docker/bentoml/requirements.txt .
RUN $VIRTUAL_ENV/bin/pip install --no-cache-dir -r requirements.txt
RUN $VIRTUAL_ENV/bin/python -m spacy download fr_core_news_sm

# expose python/pip du venv dans /usr/local/bin (déjà dans PATH minimal)
RUN ln -sf $VIRTUAL_ENV/bin/python /usr/local/bin/python \
 && ln -sf $VIRTUAL_ENV/bin/pip /usr/local/bin/pip \
 && ln -sf $VIRTUAL_ENV/bin/python3 /usr/local/bin/python3

# Copie des utilitaires
COPY src/model_serving /model_serving
COPY src/data_module_df /model_serving/src/data_module_df
COPY src/models_module_df /model_serving/src/models_module_df
COPY src/utils /model_serving/src/utils
RUN chmod +x /model_serving/model_building.sh

RUN addgroup -g 1000 trainusr \
    && adduser -D -u 1000 -G trainusr -h /home/trainusr trainusr

RUN chown trainusr /model_serving
USER trainusr

ARG WORKDIR=/app
WORKDIR ${WORKDIR}

ENV PYTHONPATH="/"
