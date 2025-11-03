FROM python:3.12-slim

WORKDIR /workspace

COPY dockerfiles/trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 trainusr
USER trainusr