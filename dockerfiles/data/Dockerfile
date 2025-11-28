FROM python:3.12-slim

WORKDIR /app

COPY dockerfiles/data/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger le modèle spaCy français
RUN python -m spacy download fr_core_news_sm

RUN mkdir -p /src && touch /src/__init__.py
COPY src/data_module_df /src/data_module_df
COPY src/pipeline_steps_df/data /src
COPY src/utils /src/utils

ENV PYTHONPATH="/"

COPY dockerfiles/data/service.py .

RUN useradd -m -u 1000 trainusr
USER trainusr

EXPOSE 8001

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]