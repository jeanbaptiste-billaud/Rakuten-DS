import streamlit as st
from pathlib import Path
import requests

# Configuration de la page
st.set_page_config(
    page_title="Rakuten MLOps Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
css_path = Path('../../docker/streamlit/assets/styles.css')
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    logo_path = Path("../../docker/streamlit/assets/logo.png")
    if logo_path.exists():
        st.image("assets/logo.png", width=300)
    st.title("🏭 Rakuten MLOps")
    st.markdown("---")

    
    st.subheader("🔗 Services")
    
    # Liens vers les services
    st.markdown("🏠 **Accueil** (page actuelle)")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("📊")
    with col2:
        st.markdown("[MLflow](http://localhost:5000)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("📈")
    with col2:
        st.markdown("[Grafana](http://localhost:3000)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("🔍")
    with col2:
        st.markdown("[Prometheus](http://localhost:9090)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("⚙️")
    with col2:
        st.markdown("[Airflow](http://localhost:8080)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("🎯")
    with col2:
        st.markdown("[Drift Detector](http://localhost:8003)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("🚀")
    with col2:
        st.markdown("[API Gateway](http://localhost:8000/docs)")


# Contenu principal
st.title("🏭 Rakuten MLOps - Dashboard Unifié")

st.markdown("""
Bienvenue sur le tableau de bord centralisé du projet Rakuten MLOps.
Cette interface vous permet d'accéder à tous les outils de la stack MLOps depuis un seul endroit.
""")

# Statut des services
st.header("📡 Statut des Services")

col1, col2, col3, col4, col5 = st.columns(5)

services = {
    "MLflow": ("http://mlflow-server:5000", col1),
    "Grafana": ("http://grafana:3000/api/health", col2),
    "Prometheus": ("http://prometheus:9090/-/healthy", col3),
    "Airflow": ("http://airflow-apiserver:8080/api/v2/version", col4),
    "Model Serving": ("http://model-serving:3001/healthz", col5)
}

for name, (url, col) in services.items():
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            col.success(f"✅ {name}")
        else:
            col.error(f"❌ {name}")
    except:
        col.warning(f"⚠️ {name}")

st.markdown("---")

# Lecture et affichage du README simplifié
st.header("📖 Documentation")

readme_path = Path("../../docker/streamlit/README_simplified.md")
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    
    # Affichage direct du markdown (Streamlit gère nativement les blocs de code)
    st.markdown(readme_content)
else:
    st.error("Documentation non disponible")

# Métriques rapides
st.header("📊 Métriques en Temps Réel")

col1, col2, col3 = st.columns(3)

# Exemple de récupération de métriques depuis Prometheus
try:
    prom_url = "http://prometheus:9090/api/v1/query"
    
    # Total de prédictions
    response = requests.get(
        prom_url,
        params={"query": "sum(model_requests_total)"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            total_preds = int(float(data['data']['result'][0]['value'][1]))
            col1.metric("Total Prédictions", f"{total_preds:,}")
        else:
            col1.metric("Total Prédictions", "N/A")
    
    # Confiance moyenne
    response = requests.get(
        prom_url,
        params={"query": "sum(rate(model_prediction_confidence_sum[5m])) / sum(rate(model_prediction_confidence_count[5m]))"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            avg_conf = float(data['data']['result'][0]['value'][1])
            col2.metric("Confiance Moyenne", f"{avg_conf:.2%}")
        else:
            col2.metric("Confiance Moyenne", "N/A")
    
    # Entropie (drift indicator)
    response = requests.get(
        prom_url,
        params={"query": "model_entropy_last_value"},
        timeout=2
    )
    if response.status_code == 200:
        data = response.json()
        if data['data']['result']:
            entropy = float(data['data']['result'][0]['value'][1])
            col3.metric("Entropie (Drift)", f"{entropy:.3f}")
        else:
            col3.metric("Entropie (Drift)", "N/A")
        
except Exception as e:
    col1.metric("Total Prédictions", "N/A")
    col2.metric("Confiance Moyenne", "N/A")
    col3.metric("Entropie (Drift)", "N/A")
    st.warning("⚠️ Impossible de récupérer les métriques Prometheus")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Rakuten MLOps Dashboard v1.0 | Propulsé par Streamlit</p>
</div>
""", unsafe_allow_html=True)
