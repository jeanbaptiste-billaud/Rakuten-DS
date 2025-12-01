FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances
COPY docker/model-serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le fichier service.py dans /app
COPY docker/model-serving/service.py /app/service.py

# Exposer le port du service model-serving
EXPOSE 8002

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8002"]
