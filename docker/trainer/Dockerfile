FROM python:3.12-slim

COPY docker/trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git

RUN mkdir -p /src && touch /src/__init__.py
COPY src/pipeline_steps_df/training /src
COPY src/pipeline_steps_df/model_promotion_decision.py /src/model_promotion_decision.py
COPY src/models_module_df /src/models_module_df
COPY src/utils /src/utils

ENV PYTHONPATH="/"

RUN useradd -m -u 1000 trainusr
USER trainusr

ARG WORKDIR=/app
WORKDIR ${WORKDIR}