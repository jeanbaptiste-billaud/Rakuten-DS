# train_text_model.py
import json
import os
import mlflow
import requests
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

from src.utils.common_utils import get_project_root
from src.models_module_df.model_text_classifier import TextClassifier

# --- Configuration ---
ROOT_PATH = os.getenv("WORKDIR", get_project_root())
DATA_PATH = os.path.join(ROOT_PATH, "data/preprocessed/preprocessed_text.csv")
Lineage_Path = os.path.join(ROOT_PATH, "lineage/lineage.json")
run_id_path = os.path.join(ROOT_PATH, "run_id.json")

MLFLOW_EXPERIMENT_NAME = "text_classification_svm"
DRIFT_DETECTOR_URL = os.getenv("DRIFT_DETECTOR_URL", "http://localhost:8003")

for required_env in ["MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
    if not os.getenv(required_env):
        raise RuntimeError(f"{required_env} must be injected by Infisical")

# --- Initialisation MLflow ---
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    print(f"[INFO] Using MLflow tracking URI from env: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
else:
    raise RuntimeError("MLFLOW_TRACKING_URI must be injected by Infisical")

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run() as run:
    run_id = run.info.run_id

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
    signature = infer_signature(X_val, classifier.predict(X_val))
    mlflow.sklearn.log_model(classifier, signature=signature, name="rakuten_text_classifier")

    # --- Evaluation ---
    print("📊 Évaluation du modèle...")
    y_pred, _ = classifier.predict(X_val)
    metrics, cm, cr = classifier.evaluate(X_val, y_val)

    # --- Log du lineage dans MLflow ---
    mlflow.log_artifact(Lineage_Path)

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

    metrics_path = "/tmp/metrics_text.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    mlflow.log_artifact(metrics_path, "eval/")

    # --- log du classification report ---
    cr_path = "/tmp/classification_report.json"
    with open(cr_path, "w") as f:
        json.dump(cr, f, indent=2)
    mlflow.log_artifact(cr_path, "eval/")

    # --- log de la matrice de confusion ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classifier.model.classes_)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    ax.set_title("Confusion Matrix")
    mlflow.log_figure(fig, "eval/confusion_matrix.png")
    plt.close(fig)

    print(f"🔗 MLflow Run ID : {run_id}")

    # ========================================
    # 🆕 DÉTECTION DE DRIFT
    # ========================================
    print("\n🔍 Envoi des données au drift detector...")

    try:
        drift_payload = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "y_true": y_val.tolist(),
            "y_pred": y_pred.tolist(),
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_f1": float(metrics["macro_f1"])
            }
        }

        response = requests.post(
            f"{DRIFT_DETECTOR_URL}/log-evaluation",
            json=drift_payload,
            params={"drift_threshold": 0.05},
            timeout=30
        )

        if response.status_code == 200:
            drift_report = response.json()

            print("\n" + "=" * 60)
            print("📊 RAPPORT DE DRIFT")
            print("=" * 60)
            print(f"Drift détecté      : {'⚠️  OUI' if drift_report['drift_detected'] else '✅ NON'}")
            print(f"Accuracy actuelle  : {drift_report['current_accuracy']:.4f}")
            print(f"Accuracy référence : {drift_report['reference_accuracy']:.4f}")
            print(f"Dégradation        : {drift_report['accuracy_drop']:.4f} ({drift_report['accuracy_drop'] * 100:.2f}%)")
            print(f"Seuil              : {drift_report['drift_threshold']:.4f} ({drift_report['drift_threshold'] * 100:.2f}%)")
            print(f"Rapport HTML       : {drift_report['report_path']}")
            print("=" * 60 + "\n")

            # Logger dans MLflow
            mlflow.log_metrics({
                "drift_detected": 1.0 if drift_report['drift_detected'] else 0.0,
                "accuracy_drop": drift_report['accuracy_drop'],
                "reference_accuracy": drift_report['reference_accuracy']
            })

            if drift_report['drift_detected']:
                print("⚠️  ALERTE : Drift de performance détecté !")
                print("   → Considérer un réentraînement du modèle")
                print("   → Vérifier les changements dans les données")
        else:
            print(f"⚠️  Erreur lors de la détection de drift: {response.status_code}")
            print(f"   Message: {response.text}")

    except requests.exceptions.ConnectionError:
        print("⚠️  Service drift-detector non disponible")
        print("   Le training continue sans détection de drift")
    except Exception as e:
        print(f"⚠️  Erreur lors de la communication avec drift-detector: {e}")
        print("   Le training continue sans détection de drift")

    print("\n✅ Entraînement et évaluation terminés avec succès")

run_id_path = "/tmp/run_id.json"
with open(run_id_path, "w") as f:
    json.dump({"run_id":run_id}, f, indent=2)
