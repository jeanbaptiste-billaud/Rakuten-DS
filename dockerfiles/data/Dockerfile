FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances
COPY dockerfiles/data/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger le modèle spaCy FR
RUN python -m spacy download fr_core_news_sm

# Copier ton code source dans l'image
RUN mkdir -p /src && touch /src/__init__.py
COPY src/data_module_df /src/data_module_df
COPY src/pipeline_steps_df/data /src
COPY src/utils /src/utils

ENV PYTHONPATH="/"

# Copier ton service FastAPI
COPY dockerfiles/data/service.py .

# Créer un utilisateur sans privilèges
RUN useradd -m -u 1000 trainusr
USER trainusr

EXPOSE 8001

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]
