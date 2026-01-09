"""
Service de détection de drift avec Evidently.
Monitore les performances du modèle au fil des réentraînements.
"""

import os
import httpx
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Chemins de stockage
STORAGE_DIR = Path("/app/drift_reports")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = STORAGE_DIR / "reference_data.csv"
HISTORY_PATH = STORAGE_DIR / "metrics_history.json"

# ========================================
# MODEL SERVING CONFIG
# ========================================

MODEL_SERVING_URL = os.getenv(
    "MODEL_SERVING_URL",
    "http://model-serving:8002"
)

# ============================
# DATA LINEAGE
# ============================

DATA_DIR = Path("/app/data")
DATA_LINEAGE_DIR = DATA_DIR / "data_lineage_prediction"
DATA_LINEAGE_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_LINEAGE_PATH = DATA_LINEAGE_DIR / "prediction_lineage.csv"

# ============================
# FASTAPI APP
# ============================

app = FastAPI(
    title="Rakuten Drift Detector Service",
    description="Service de détection de drift avec Evidently AI",
    version="1.0.0"
)

# ============================
# SCHEMAS
# ============================

class EvaluationData(BaseModel):
    """Données d'évaluation après entraînement."""
    run_id: str = Field(..., description="ID du run MLflow")
    timestamp: str = Field(..., description="Timestamp ISO format")
    y_true: list = Field(..., description="Vrais labels")
    y_pred: list = Field(..., description="Prédictions")
    metrics: Dict[str, float] = Field(..., description="Métriques (accuracy, f1, etc.)")


class DriftReport(BaseModel):
    """Résultat de l'analyse de drift."""
    drift_detected: bool
    current_accuracy: float
    reference_accuracy: float
    accuracy_drop: float
    drift_threshold: float
    report_path: str
    run_id: Optional[str] = None


# ============================
# AJOUT — PREDICT SCHEMA
# ============================

class PredictRequest(BaseModel):
    """Requête de prédiction en ligne."""
    text_cleaned: str = Field(
        ..., description="Texte nettoyé en entrée du modèle"
    )


# ========================================
# GESTION DES DONNÉES DE RÉFÉRENCE
# ========================================

def save_reference_data(y_true: list, y_pred: list, run_id: str) -> None:
    """Sauvegarde les données de référence (baseline)."""
    df = pd.DataFrame({
        'target': y_true,
        'prediction': y_pred,
        'run_id': run_id
    })
    df.to_csv(REFERENCE_PATH, index=False)
    logger.info(f"✅ Données de référence sauvegardées (run: {run_id})")


def load_reference_data() -> Optional[pd.DataFrame]:
    """Charge les données de référence."""
    if not REFERENCE_PATH.exists():
        return None
    return pd.read_csv(REFERENCE_PATH)


def save_metrics_history(eval_data: EvaluationData, drift_report: DriftReport) -> None:
    """Ajoute les métriques et infos de drift à l'historique."""
    history = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)

    history.append({
        'run_id': eval_data.run_id,
        'timestamp': eval_data.timestamp,
        'metrics': eval_data.metrics,
        'drift_detected': drift_report.drift_detected,
        'accuracy_drop': drift_report.accuracy_drop,
        'reference_accuracy': drift_report.reference_accuracy,
        'current_accuracy': drift_report.current_accuracy
    })

    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

# ========================================
# AJOUT — DATA LINEAGE
# ========================================

def save_prediction_lineage(
    lineage_id: str,
    timestamp: str,
    model_version: str,
    features: Dict,
    prediction: int,
    confidence: float
) -> None:
    """Sauvegarde une prédiction et son data lineage."""

    row = {
        "lineage_id": lineage_id,
        "timestamp": timestamp,
        "model_version": model_version,
        "features": json.dumps(features),
        "prediction": prediction,
        "confidence": confidence
    }

    df = pd.DataFrame([row])

    if PREDICTION_LINEAGE_PATH.exists():
        df.to_csv(PREDICTION_LINEAGE_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(PREDICTION_LINEAGE_PATH, index=False)

    logger.info(f"🧬 Lineage sauvegardé (id={lineage_id})")

# ========================================
# GÉNÉRATION DE RAPPORTS EVIDENTLY
# ========================================

def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    run_id: str
) -> str:
    """Génère un rapport Evidently HTML."""

    column_mapping = ColumnMapping(
        target='target',
        prediction='prediction'
    )

    report = Report(metrics=[
        ClassificationPreset()
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )

    report_path = STORAGE_DIR / f"drift_report_{run_id}.html"
    report.save_html(str(report_path))

    logger.info(f"📊 Rapport Evidently généré : {report_path}")
    return str(report_path)


def detect_performance_drift(
    eval_data: EvaluationData,
    threshold: float = 0.05
) -> DriftReport:
    """
    Détecte le drift de performance.
    
    Args:
        eval_data: Données d'évaluation actuelles
        threshold: Seuil de dégradation acceptable (default: 5%)
    
    Returns:
        Rapport de drift avec métriques
    """
    # Charger les données de référence
    reference_df = load_reference_data()
    
    if reference_df is None:
        # Première exécution : définir comme référence
        save_reference_data(eval_data.y_true, eval_data.y_pred, eval_data.run_id)
        logger.info("🎯 Première exécution : baseline définie")
        
        return DriftReport(
            drift_detected=False,
            current_accuracy=eval_data.metrics['accuracy'],
            reference_accuracy=eval_data.metrics['accuracy'],
            accuracy_drop=0.0,
            drift_threshold=threshold,
            report_path="N/A (baseline)",
            run_id=eval_data.run_id
        )
    
    # Créer DataFrame pour les données actuelles
    current_df = pd.DataFrame({
        'target': eval_data.y_true,
        'prediction': eval_data.y_pred,
        'run_id': eval_data.run_id
    })
    
    # Générer rapport Evidently
    report_path = generate_drift_report(reference_df, current_df, eval_data.run_id)
    
    # Calculer la dégradation
    reference_accuracy = (reference_df['target'] == reference_df['prediction']).mean()
    current_accuracy = eval_data.metrics['accuracy']
    accuracy_drop = reference_accuracy - current_accuracy
    
    drift_detected = accuracy_drop > threshold
    
    if drift_detected:
        logger.warning(
            f"⚠️ DRIFT DÉTECTÉ ! "
            f"Dégradation de {accuracy_drop:.2%} "
            f"(seuil: {threshold:.2%})"
        )
    else:
        logger.info(
            f"✅ Pas de drift. "
            f"Différence: {accuracy_drop:.2%} "
            f"(seuil: {threshold:.2%})"
        )
    
    return DriftReport(
        drift_detected=drift_detected,
        current_accuracy=current_accuracy,
        reference_accuracy=reference_accuracy,
        accuracy_drop=accuracy_drop,
        drift_threshold=threshold,
        report_path=report_path,
        run_id=eval_data.run_id
    )


# ========================================
# ENDPOINTS API
# ========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil avec liste des rapports de drift."""
    
    # Charger l'historique
    history = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, 'r') as f:
            history = json.load(f)
    
    # Trier par timestamp (plus récent en premier)
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Générer les lignes HTML pour chaque rapport
    rows_html = ""
    for i, entry in enumerate(history, 1):
        run_id = entry['run_id']
        timestamp = entry['timestamp']
        metrics = entry['metrics']
        
        # Formater la date
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = timestamp
        
        # Récupérer les infos de drift
        accuracy = entry.get('current_accuracy', metrics.get('accuracy', 0.0))
        drift_detected = entry.get('drift_detected', False)
        accuracy_drop = entry.get('accuracy_drop', 0.0)
        reference_accuracy = entry.get('reference_accuracy', accuracy)
        
        # Affichage du drift
        if drift_detected:
            drift_display = "⚠️ OUI"
            drift_class = "drift-yes"
            drift_info = f"({accuracy_drop*100:.2f}%)"
        else:
            drift_display = "✅ NON"
            drift_class = "drift-no"
            drift_info = ""
        
        # Lien vers le rapport
        report_link = f"/report/{run_id}"
        
        rows_html += f"""
        <tr>
            <td>{i}</td>
            <td>{date_str}</td>
            <td><a href="{report_link}" class="run-id">{run_id[:12]}...</a></td>
            <td>{accuracy:.4f}</td>
            <td class="{drift_class}">{drift_display} {drift_info}</td>
            <td><a href="{report_link}" class="btn-view">📊 Voir le rapport</a></td>
        </tr>
        """
    
    # Si aucun rapport
    if not rows_html:
        rows_html = """
        <tr>
            <td colspan="6" style="text-align: center; padding: 40px; color: #666;">
                Aucun rapport de drift disponible.<br>
                Lancez un entraînement pour créer le premier rapport.
            </td>
        </tr>
        """
    
    # Référence existe ?
    reference_status = "✅ Définie" if REFERENCE_PATH.exists() else "⚠️ Non définie"
    reference_class = "status-ok" if REFERENCE_PATH.exists() else "status-warning"
    
    # Total de rapports
    total_reports = len(history)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Drift Detector - Rakuten MLOps</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .stats {{
                display: flex;
                justify-content: space-around;
                padding: 30px;
                background: #f8f9fa;
                border-bottom: 2px solid #e9ecef;
            }}
            
            .stat-box {{
                text-align: center;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                min-width: 200px;
            }}
            
            .stat-box .label {{
                font-size: 0.9em;
                color: #6c757d;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .stat-box .value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}
            
            .status-ok {{
                color: #28a745 !important;
            }}
            
            .status-warning {{
                color: #ffc107 !important;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section-title {{
                font-size: 1.8em;
                margin-bottom: 20px;
                color: #333;
                border-left: 5px solid #667eea;
                padding-left: 15px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-radius: 10px;
                overflow: hidden;
            }}
            
            thead {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            
            th {{
                padding: 20px;
                text-align: left;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 1px;
            }}
            
            td {{
                padding: 18px 20px;
                border-bottom: 1px solid #e9ecef;
                color: #333;
            }}
            
            tr:last-child td {{
                border-bottom: none;
            }}
            
            tbody tr:hover {{
                background-color: #f8f9fa;
                transition: background-color 0.3s ease;
            }}
            
            .run-id {{
                font-family: 'Courier New', monospace;
                background: #f8f9fa;
                padding: 5px 10px;
                border-radius: 5px;
                color: #667eea;
                text-decoration: none;
                font-weight: bold;
            }}
            
            .run-id:hover {{
                background: #667eea;
                color: white;
            }}
            
            .drift-yes {{
                color: #dc3545;
                font-weight: bold;
            }}
            
            .drift-no {{
                color: #28a745;
                font-weight: bold;
            }}
            
            .btn-view {{
                display: inline-block;
                padding: 8px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-size: 0.9em;
                transition: all 0.3s ease;
                box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
            }}
            
            .btn-view:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.6);
            }}
            
            .footer {{
                text-align: center;
                padding: 30px;
                background: #f8f9fa;
                color: #6c757d;
                border-top: 2px solid #e9ecef;
            }}
            
            .footer a {{
                color: #667eea;
                text-decoration: none;
                font-weight: bold;
            }}
            
            .footer a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Drift Detector Service</h1>
                <p>Monitoring de la performance des modèles ML avec Evidently AI</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="label">Total Rapports</div>
                    <div class="value">{total_reports}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Baseline</div>
                    <div class="value {reference_class}">{reference_status}</div>
                </div>
                <div class="stat-box">
                    <div class="label">Framework</div>
                    <div class="value" style="font-size: 1.5em;">Evidently AI</div>
                </div>
            </div>
            
            <div class="content">
                <h2 class="section-title">📊 Historique des Rapports de Drift</h2>
                
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Date / Heure</th>
                            <th>Run ID</th>
                            <th>Accuracy</th>
                            <th>Drift Détecté</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>
                    <strong>Rakuten MLOps Platform</strong> • 
                    <a href="/health">Health Check</a> • 
                    <a href="/history">API History</a> • 
                    <a href="http://localhost:5000" target="_blank">MLflow UI</a>
                </p>
                <p style="margin-top: 10px; font-size: 0.9em;">
                    Propulsé par FastAPI + Evidently AI
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Vérifie l'état de santé du service."""
    return {
        "status": "healthy",
        "storage_dir": str(STORAGE_DIR),
        "reference_exists": REFERENCE_PATH.exists()
    }

@app.post("/predict")
async def predict(request: PredictRequest):

    lineage_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MODEL_SERVING_URL}/predict",
            json={"text_cleaned": request.text_cleaned},
            timeout=10.0
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail="Erreur lors de l'appel au model-serving"
        )

    result = response.json()

    prediction = result["predicted_class"]
    confidence = result["confidence"]

    save_prediction_lineage(
        lineage_id=lineage_id,
        timestamp=timestamp,
        model_version="from-model-serving",
        features={"text_cleaned": request.text_cleaned},
        prediction=prediction,
        confidence=confidence
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "lineage_id": lineage_id
    }



@app.post("/log-evaluation", response_model=DriftReport)
async def log_evaluation(eval_data: EvaluationData, drift_threshold: float = 0.05):
    """
    Enregistre une évaluation et détecte le drift.
    
    Args:
        eval_data: Données d'évaluation (labels, prédictions, métriques)
        drift_threshold: Seuil de dégradation (default: 5%)
    
    Returns:
        Rapport de drift
    """
    try:
        logger.info(f"📥 Réception évaluation (run: {eval_data.run_id})")
        
        # Détecter le drift
        drift_report = detect_performance_drift(eval_data, drift_threshold)
        
        # Sauvegarder dans l'historique (avec les infos de drift)
        save_metrics_history(eval_data, drift_report)
        
        return drift_report
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de drift: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de détection de drift: {str(e)}"
        )


@app.get("/report/{run_id}")
async def get_report(run_id: str):
    """Récupère le rapport HTML pour un run donné."""
    report_path = STORAGE_DIR / f"drift_report_{run_id}.html"
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Rapport introuvable pour le run {run_id}"
        )
    
    with open(report_path, 'r') as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@app.get("/history")
async def get_history():
    """Récupère l'historique complet des métriques."""
    if not HISTORY_PATH.exists():
        return {"history": []}
    
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
    
    return {"history": history}


@app.post("/reset-baseline")
async def reset_baseline(eval_data: EvaluationData):
    """Redéfinit la baseline de référence."""
    try:
        save_reference_data(eval_data.y_true, eval_data.y_pred, eval_data.run_id)
        return {
            "status": "success",
            "message": f"Baseline redéfinie avec le run {eval_data.run_id}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la redéfinition de la baseline: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
