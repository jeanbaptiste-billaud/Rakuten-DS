# train_text_model_mlflow.py
import json
import os

import bentoml
import mlflow.sklearn
import pandas as pd
from bentoml.types import ModelSignature
from sklearn.model_selection import train_test_split

from src.common_utils import get_project_root
from src.models_module_df.model_text_classifier import TextClassifier

# --- Configuration ---
ROOT_PATH = get_project_root()
DATA_PATH = os.path.join(ROOT_PATH, "data/preprocessed/preprocessed_text.csv")

OUTPUT_DIR = os.path.join(ROOT_PATH, "models/text_classifier")
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    classifier = TextClassifier()
    classifier.train(X_train, y_train)

    # --- Evaluation ---
    metrics, cm, cr = classifier.evaluate(X_val, y_val)

    # --- Sauvegarde du modèle avec BentoML ---
    bento_model = bentoml.sklearn.save_model(
        "text_classifier_svm",
        classifier.model,
        signatures={
            "predict": ModelSignature(),
            "predict_proba": ModelSignature(),
        },
        metadata={
            "framework": "scikit-learn",
            "source": "MLflow-tracked",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"]
        }
    )
    print(f"📦 Modèle enregistré dans BentoML : {bento_model.tag}")

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

    # --- le tag du modèle Bento est enregistré dans mlflow
    mlflow.log_text(str(bento_model.tag), "bentoml_model_tag.txt")

    print(f"📊 Métriques : {metrics}")
    print(f"🔗 MLflow Run ID : {run.info.run_id}")
