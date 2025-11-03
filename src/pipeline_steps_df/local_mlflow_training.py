# local_mlflow_training.py
import json
import os
import mlflow

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common_utils import get_project_root
from src.models_module_df.model_text_classifier import TextClassifier

# --- Configuration ---
ROOT_PATH = get_project_root()
DATA_PATH = os.path.join(ROOT_PATH, "data/preprocessed/preprocessed_text.csv")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("text_classification_svm")

# --- Initialisation MLflow ---
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    print(f"[INFO] Using MLflow tracking URI from env: {mlflow_uri}")
else:
    # Fallback par défaut (utile pour exécution locale)
    mlflow_uri = "http://localhost:5000"
    print(f"[INFO] MLFLOW_TRACKING_URI not set — using default: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)

with mlflow.start_run(run_name="SVM_TFIDF_TextClassifier") as run:
    run_id = run.info.run_id
    OUTPUT_DIR = os.path.join(ROOT_PATH, f"models/text_classifier/{run_id}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_text.json")

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

    # --- Entraînement et sauvegarde du modèle ---
    print("🚀 Entraînement du modèle de classification textuelle...")
    classifier = TextClassifier()
    classifier.train(X_train, y_train)
    mlflow.sklearn.save_model(classifier, OUTPUT_DIR)

    # --- Evaluation ---
    metrics, cm, cr = classifier.evaluate(X_val, y_val)

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
    mlflow.log_metrics(metrics)  # attends un dictionnaire

    # --- Sauvegarde des résultats d'évaluation et enregistrement de l'emplacement du fichier dans mlflow ---
    metrics["confusion_matrix"] = cm
    metrics["classification_report"] = cr
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.log_artifact(METRICS_PATH, artifact_path="metrics")  # attends fichiers (JSON, CSV, image, modèle)

    print(f"🔗 MLflow Run ID : {run_id}")
