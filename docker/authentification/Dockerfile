# docker/authentification/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances communes
COPY docker/authentification/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers d'authentification
COPY docker/authentification/auth.json /auth.json
COPY docker/authentification/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copier le module Python d’authentification
COPY docker/authentification/auth_utils.py /auth_utils.py

# Ajouter le répertoire src au PYTHONPATH (pour les imports Python)
ENV PYTHONPATH=/app

# Entrypoint commun : vérifie le token avant d’exécuter quoi que ce soit
ENTRYPOINT ["/entrypoint.sh"]
