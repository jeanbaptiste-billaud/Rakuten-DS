#!/bin/sh

# Désactive l'arrêt sur erreur
set +e

docker run --rm \
  -v mlflow_db:/db_data \
  -v ../data:/backup \
  alpine \
   sh -c "tar czvf /backup/mlflow_db.tar.gz -C /db_data . && chown $(id -u):$(id -g) /backup/mlflow_db.tar.gz"