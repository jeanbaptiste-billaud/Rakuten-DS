FROM python:3.12-slim

WORKDIR /app

# Dépendances Python
COPY docker/auth_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code du service d'auth
COPY docker/auth_service/service.py app.py
COPY docker/auth_service/auth_utils.py .
COPY docker/auth_service/auth.json .

EXPOSE 8000

# Lancement du service HTTP
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
