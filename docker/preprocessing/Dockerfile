FROM python:3.12-slim

WORKDIR /app

COPY docker/preprocessing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger le modèle spaCy français
RUN python -m spacy download fr_core_news_sm

COPY docker/preprocessing/service.py .

EXPOSE 8001

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]