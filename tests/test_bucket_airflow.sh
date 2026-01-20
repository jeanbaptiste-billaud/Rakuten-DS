#!/bin/bash
set -e

# --------------------------------------------
# ⚙️ Configuration
# --------------------------------------------
MINIO_HOST="minio"
MINIO_PORT="9000"
MINIO_BUCKET="mlflow-artifacts-bucket"
AWS_ACCESS_KEY_ID="minio"
AWS_SECRET_ACCESS_KEY="minio123"
TEST_KEY="test_upload/hello.txt"
TEST_BODY="hello from mlflow test script"

# --------------------------------------------
# 🐳 Exécution du test depuis le container MLflow
# --------------------------------------------
echo "🚀 Lancement du test d'écriture MinIO depuis le container MLflow..."

docker compose exec -T mlflow bash -c "
  python3 - <<'EOF'
import boto3, sys

endpoint = 'http://${MINIO_HOST}:${MINIO_PORT}'
bucket = '${MINIO_BUCKET}'
key = '${TEST_KEY}'
body = b'${TEST_BODY}'
access = '${AWS_ACCESS_KEY_ID}'
secret = '${AWS_SECRET_ACCESS_KEY}'

print(f'🔗 Connexion à {endpoint}...')
s3 = boto3.client(
    's3',
    endpoint_url=endpoint,
    aws_access_key_id=access,
    aws_secret_access_key=secret
)

# Vérifie que le bucket existe
buckets = [b['Name'] for b in s3.list_buckets()['Buckets']]
if bucket not in buckets:
    print(f'⚠️  Le bucket {bucket} n\'existe pas, création...')
    s3.create_bucket(Bucket=bucket)

# Upload du fichier test
print(f'📤 Envoi de l\'objet s3://{bucket}/{key}...')
s3.put_object(Bucket=bucket, Key=key, Body=body)
print('✅ Upload réussi !')

# Vérifie la présence de l'objet
resp = s3.list_objects_v2(Bucket=bucket, Prefix=key)
if resp.get('KeyCount', 0) > 0:
    print('🧾 Fichier présent sur MinIO :', resp['Contents'][0]['Key'])
else:
    print('❌ L\'objet n\'a pas été trouvé après upload.')

EOF
"

echo "🎉 Test terminé : vérifie sur l'UI MinIO (http://localhost:9000) dans le bucket '${MINIO_BUCKET}'."
