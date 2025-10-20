# train_text_model_mlflow.py
import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from src.models_module_df.model_text_classifier import TextClassifier
from src.common_utils import get_project_root

# --- Configuration ---
ROOT_PATH = get_project_root()
DATA_PATH = os.path.join(ROOT_PATH, "data/preprocessed/preprocessed_text.csv")
OUTPUT_DIR = os.path.join(ROOT_PATH, "models/text_classifier")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "text_model.joblib")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_text.json")

MLFLOW_EXPERIMENT_NAME = "text_classification_svm"

# --- Initialisation MLflow ---
mlflow.set_tracking_uri("http://localhost:5000")  # ou ton endpoint Dagshub
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name="SVM_TFIDF_TextClassifier") as run:
    print("📂 Chargement des données prétraitées...")
    df = pd.read_csv(DATA_PATH)

    if "text_cleaned" not in df.columns or "prdtypecode" not in df.columns:
        raise ValueError("❌ Le dataset doit contenir les colonnes 'text_cleaned' et 'prdtypecode'.")

    X = df["text_cleaned"].astype(str)
    y = df["prdtypecode"]

    # --- Split train/val ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Entraînement ---
    print("🚀 Entraînement du modèle de classification textuelle...")
    classifier = TextClassifier(MODEL_PATH)
    classifier.train(X_train, y_train)

    # --- Prédictions ---
    preds, probs = classifier.predict(X_val)
    metrics = classifier.evaluate(y_val, preds, probs)

    # --- Log des paramètres dans MLflow ---
    mlflow.log_params({
        "vectorizer": "TfidfVectorizer(max_features=45000)",
        "model_type": "SVM-RBF",
        "C": 12,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
        "random_state": 42,
        "test_size": 0.2
    })

    # --- Log des métriques ---
    mlflow.log_metrics(metrics)

    # --- Sauvegarde locale du modèle ---
    classifier.save()
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Log artefacts ---
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    mlflow.log_artifact(METRICS_PATH, artifact_path="metrics")

    # --- Sauvegarde du modèle dans MLflow ---
    mlflow.sklearn.log_model(classifier.model, artifact_path="model_sklearn")

    print(f"✅ Modèle sauvegardé dans {MODEL_PATH}")
    print(f"📊 Métriques : {metrics}")
    print(f"🔗 MLflow Run ID : {run.info.run_id}")