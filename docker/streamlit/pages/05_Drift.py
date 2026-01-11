import streamlit as st
import os
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Drift Detection - Rakuten MLOps",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Drift Detection - Surveillance du Modèle")

st.markdown("""
Le système de détection de drift surveille la qualité et la stabilité du modèle 
en analysant les distributions de données et les prédictions.
""")

# Récupération de l'URL depuis les variables d'environnement
drift_url = os.getenv("DRIFT_DETECTOR_URL", "http://drift-detector:8003")

# Statut du service
st.markdown("### 📡 Statut du Service")

col1, col2, col3 = st.columns(3)

try:
    response = requests.get(f"{drift_url}/health", timeout=2)
    if response.status_code == 200:
        col1.success("✅ Service Actif")
        health_data = response.json()
        col2.info(f"⏱️ Uptime: {health_data.get('uptime', 'N/A')}")
        col3.info(f"📊 Rapports: {health_data.get('total_reports', 'N/A')}")
    else:
        col1.error("❌ Service Inactif")
except:
    col1.warning("⚠️ Service Inaccessible")

st.markdown("---")

# Rapports de drift disponibles
st.markdown("### 📂 Rapports de Drift Disponibles")

reports_dir = Path("/app/reports/drift_reports")

if reports_dir.exists():
    # Liste des rapports HTML
    html_reports = sorted(
        reports_dir.glob("*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if html_reports:
        st.success(f"✅ {len(html_reports)} rapport(s) trouvé(s)")
        
        # Sélection du rapport
        report_names = [f.name for f in html_reports]
        selected_report = st.selectbox(
            "Choisir un rapport à afficher :",
            report_names,
            index=0
        )
        
        # Affichage du rapport sélectionné
        if selected_report:
            report_path = reports_dir / selected_report
            
            # Informations sur le rapport
            col1, col2, col3 = st.columns(3)
            
            file_stat = report_path.stat()
            file_size = file_stat.st_size / 1024  # En KB
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
            
            col1.metric("Nom", selected_report[:30] + "..." if len(selected_report) > 30 else selected_report)
            col2.metric("Taille", f"{file_size:.2f} KB")
            col3.metric("Dernière Modification", file_mtime.strftime("%Y-%m-%d %H:%M"))
            
            st.markdown("---")
            
            # Affichage du rapport HTML
            st.markdown("### 📊 Rapport Evidently")
            
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Affichage en iframe
                st.components.v1.html(html_content, height=1000, scrolling=True)
                
                # Bouton de téléchargement
                st.download_button(
                    label="📥 Télécharger le rapport",
                    data=html_content,
                    file_name=selected_report,
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du rapport: {str(e)}")
        
        # Liste complète des rapports
        with st.expander("📋 Liste complète des rapports"):
            reports_data = []
            for report in html_reports:
                file_stat = report.stat()
                reports_data.append({
                    'Nom': report.name,
                    'Taille (KB)': f"{file_stat.st_size / 1024:.2f}",
                    'Date': datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
            
            df_reports = pd.DataFrame(reports_data)
            st.dataframe(df_reports, use_container_width=True)
    else:
        st.info("📭 Aucun rapport de drift disponible pour le moment")
        st.markdown("""
        Les rapports seront générés automatiquement par le service de détection de drift 
        lors de l'analyse des prédictions du modèle.
        """)
else:
    st.warning(f"⚠️ Répertoire des rapports introuvable: {reports_dir}")
    st.info("""
    Assurez-vous que le volume `logs_and_reports` est correctement monté 
    et que le service drift-detector a généré des rapports.
    """)

# Métriques de drift en temps réel
st.markdown("---")
st.markdown("### 📈 Métriques de Drift en Temps Réel")

col1, col2, col3, col4 = st.columns(4)

try:
    # Récupération des métriques depuis Prometheus
    prom_url = "http://prometheus:9090/api/v1/query"
    
    # Entropie (indicateur de drift)
    response = requests.get(
        prom_url,
        params={"query": "model_entropy_last_value"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            entropy = float(data['data']['result'][0]['value'][1])
            col1.metric("Entropie", f"{entropy:.3f}")
            
            # Indicateur de drift basé sur l'entropie
            if entropy > 2.5:
                col1.error("⚠️ Drift Élevé")
            elif entropy > 2.0:
                col1.warning("⚠️ Drift Modéré")
            else:
                col1.success("✅ Stable")
        else:
            col1.metric("Entropie", "N/A")
    
    # Confiance moyenne des prédictions
    response = requests.get(
        prom_url,
        params={"query": "sum(rate(model_prediction_confidence_sum[5m])) / sum(rate(model_prediction_confidence_count[5m]))"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            confidence = float(data['data']['result'][0]['value'][1])
            col2.metric("Confiance Moy.", f"{confidence:.2%}")
            
            if confidence < 0.7:
                col2.warning("⚠️ Confiance Faible")
            else:
                col2.success("✅ Confiance OK")
        else:
            col2.metric("Confiance Moy.", "N/A")
    
    # Distribution des classes
    response = requests.get(
        prom_url,
        params={"query": "count by (predicted_class) (model_requests_total)"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            num_classes = len(data['data']['result'])
            col3.metric("Classes Actives", num_classes)
        else:
            col3.metric("Classes Actives", "N/A")
    
    # Total de prédictions (dernière heure)
    response = requests.get(
        prom_url,
        params={"query": "increase(model_requests_total[1h])"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            total = int(float(data['data']['result'][0]['value'][1]))
            col4.metric("Prédictions (1h)", f"{total:,}")
        else:
            col4.metric("Prédictions (1h)", "0")
    
except Exception as e:
    col1.metric("Entropie", "N/A")
    col2.metric("Confiance Moy.", "N/A")
    col3.metric("Classes Actives", "N/A")
    col4.metric("Prédictions (1h)", "N/A")
    st.warning("⚠️ Impossible de récupérer les métriques Prometheus")

# Informations utiles
with st.expander("ℹ️ Aide - Drift Detection"):
    st.markdown("""
    ### Qu'est-ce que le drift ?
    
    Le **drift** (dérive) survient quand la distribution des données en production 
    diffère de celle des données d'entraînement, affectant les performances du modèle.
    
    **Types de drift :**
    
    1. **Data Drift (Covariate Shift)** :
       - Les features changent mais la relation feature-target reste stable
       - Exemple : Nouvelles catégories de produits
    
    2. **Concept Drift** :
       - La relation entre features et target change
       - Exemple : Changement dans les préférences des clients
    
    3. **Label Drift** :
       - La distribution des classes change
       - Exemple : Certaines catégories deviennent plus fréquentes
    
    ### Métriques de surveillance :
    
    - **Entropie** : Mesure l'incertitude des prédictions
      - Valeur basse (< 2.0) : Modèle confiant
      - Valeur élevée (> 2.5) : Modèle incertain → Drift possible
    
    - **Confiance Moyenne** : Probabilité moyenne de la classe prédite
      - Haute (> 0.8) : Prédictions fiables
      - Basse (< 0.7) : Prédictions incertaines
    
    - **Distribution des Classes** : Équilibre entre les catégories
      - Changement soudain → Label drift
    
    ### Rapports Evidently :
    
    Les rapports générés incluent :
    - **Data Drift Report** : Détection du drift sur les features
    - **Target Drift Report** : Analyse de la dérive des prédictions
    - **Data Quality Report** : Validation de la qualité des données
    - **Classification Performance** : Métriques de performance
    
    ### Actions en cas de drift détecté :
    
    1. **Analyser** : Identifier les features qui dérivent
    2. **Investiguer** : Comprendre la cause (données, modèle, concept)
    3. **Corriger** :
       - Ré-entraîner le modèle avec nouvelles données
       - Ajuster le preprocessing
       - Revoir les features engineering
    4. **Monitorer** : Suivre l'évolution après correction
    
    ### API Drift Detector :
    
    ```python
    import requests
    
    # Déclencher une analyse de drift
    response = requests.post(
        'http://localhost:8003/analyze_drift',
        json={
            'reference_data': [...],
            'current_data': [...]
        }
    )
    
    # Récupérer le dernier rapport
    response = requests.get('http://localhost:8003/reports/latest')
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 Drift Detection | Propulsé par Evidently AI</p>
</div>
""", unsafe_allow_html=True)
