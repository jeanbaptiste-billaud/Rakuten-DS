FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG APP_DIR=docker/streamlit

# Installation des packages Python
COPY ${APP_DIR}/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des fichiers de l'application
COPY ${APP_DIR}/app.py .
COPY ${APP_DIR}/pages/ ./pages/
COPY ${APP_DIR}/assets/ ./assets/
COPY ${APP_DIR}/README_simplified.md .

# Configuration Streamlit
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
port = 8501\n\
address = 0.0.0.0\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[theme]\n\
primaryColor = '#E63946'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#F1FAEE'\n\
textColor = '#1D3557'\n\
" > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py"]
