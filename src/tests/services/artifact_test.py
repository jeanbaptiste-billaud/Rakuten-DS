import mlflow, os, tempfile

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("debug_artifacts")

with mlflow.start_run() as run:
    p = tempfile.NamedTemporaryFile(delete=False)
    p.write(b"hello")
    p.close()
    mlflow.log_artifact(p.name, artifact_path="debug")
    os.unlink(p.name)
    print(run.info.artifact_uri, flush=True)
