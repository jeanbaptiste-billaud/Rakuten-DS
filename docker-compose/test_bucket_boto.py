import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
)

# Test listage
print(s3.list_buckets())

# Test upload direct
s3.put_object(
    Bucket="mlflow-artifacts-bucket",
    Key="test_upload/hello.txt",
    Body=b"hello from boto3"
)
