FROM python:3.12-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements
COPY docker/drift-detector/requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du service
COPY docker/drift-detector/service.py .

# Création du dossier de stockage
RUN mkdir -p /app/drift_reports

# Exposition du port
EXPOSE 8003

# Commande de démarrage
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8003"]
