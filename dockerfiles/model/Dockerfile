FROM python:3.12-slim

WORKDIR /app

COPY dockerfiles/model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dockerfiles/model/service.py .

EXPOSE 8002

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8002"]