ARG VERSION

FROM ghcr.io/mlflow/mlflow:${VERSION}

# On installe juste les dépendances
RUN pip install --no-cache-dir psycopg2-binary boto3 gunicorn
RUN mkdir -p /workspace/models && chmod -R 777 /workspace/models


# AUCUN CMD, AUCUN ENTRYPOINT ICI.
# On laisse le docker-compose décider.