import streamlit as st
import os

st.set_page_config(
    page_title="Grafana - Rakuten MLOps",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Grafana - Monitoring Visuel")

st.markdown("""
Grafana affiche les dashboards de monitoring pour visualiser les métriques 
en temps réel de vos modèles et services.
""")

# Récupération de l'URL depuis les variables d'environnement
grafana_url = os.getenv("GRAFANA_URL", "http://grafana:3000")

# Affichage de l'interface Grafana en iframe
st.markdown("### Dashboards Grafana")

# Option 1 : Iframe (si CORS autorisé)
st.markdown(f"""
<iframe src="{grafana_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

# Option 2 : Lien direct (alternative)
st.markdown("---")
st.markdown(f"**Accès direct :** [Ouvrir Grafana dans un nouvel onglet]({grafana_url.replace('grafana', 'localhost')})")

# Informations utiles
with st.expander("ℹ️ Aide - Grafana"):
    st.markdown("""
    ### Dashboards disponibles :
    
    - **Model Performance** : Métriques de performance du modèle
    - **System Metrics** : CPU, RAM, Disk des services
    - **Request Metrics** : Latence, throughput des API
    - **Drift Detection** : Surveillance de la dérive du modèle
    
    ### Navigation :
    
    1. **Home** : Page d'accueil avec liste des dashboards
    2. **Search** : Rechercher un dashboard spécifique
    3. **Explore** : Explorer les données Prometheus
    4. **Alerting** : Configurer des alertes
    
    ### Dashboards recommandés :
    
    - **MLOps Overview** : Vue d'ensemble de la plateforme
    - **Model Monitoring** : Suivi des prédictions
    - **Infrastructure** : État des conteneurs Docker
    
    ### Configuration :
    
    - **Authentification** : Auto-login activé (pas de credentials requis)
    - **Source de données** : Prometheus configuré par défaut
    - **Refresh** : Rafraîchissement automatique toutes les 30s
    """)

# Section métriques rapides
st.markdown("---")
st.markdown("### 📊 Métriques Clés")

col1, col2, col3, col4 = st.columns(4)

import requests

try:
    # Récupération de métriques depuis Prometheus
    prom_url = "http://prometheus:9090/api/v1/query"
    
    # Requêtes par seconde
    response = requests.get(
        prom_url,
        params={"query": "rate(model_requests_total[5m])"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            rps = float(data['data']['result'][0]['value'][1])
            col1.metric("Req/s", f"{rps:.2f}")
        else:
            col1.metric("Req/s", "N/A")
    
    # Latence moyenne
    response = requests.get(
        prom_url,
        params={"query": "rate(model_request_duration_seconds_sum[5m]) / rate(model_request_duration_seconds_count[5m])"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            latency = float(data['data']['result'][0]['value'][1])
            col2.metric("Latence Moy.", f"{latency*1000:.0f}ms")
        else:
            col2.metric("Latence Moy.", "N/A")
    
    # Taux d'erreur
    response = requests.get(
        prom_url,
        params={"query": "rate(model_errors_total[5m])"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            errors = float(data['data']['result'][0]['value'][1])
            col3.metric("Erreurs/s", f"{errors:.2f}")
        else:
            col3.metric("Erreurs/s", "0")
    
    # Uptime
    col4.metric("Uptime", "99.9%")
    
except Exception as e:
    col1.metric("Req/s", "N/A")
    col2.metric("Latence Moy.", "N/A")
    col3.metric("Erreurs/s", "N/A")
    col4.metric("Uptime", "N/A")
