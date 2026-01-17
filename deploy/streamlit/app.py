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

# =========================
# 🧪 Démo API / Sécurité
# =========================
st.markdown("---")
st.header("🧪 Démo API / Sécurité reverse proxy (via le token d'authentification)")

st.markdown(
    """
L'objectif est d'envoyer **la même requête** à l'API `/predict` mais avec **2 tokens différents** :
- **token = 456** : attendu *OK* (accès admin autorisé) ✅
- **token = 100** : attendu *KO* (accès refusé) 🚫 → démonstration de la sécurité
"""
)


api_url = st.text_input("URL de l'endpoint /predict", value="http://reverse-proxy/predict")
text_cleaned = st.text_input("Valeur de text_cleaned", value="chaussures de sport")


def call_predict(url: str, token: str, text_value: str) -> dict:
    """Appelle /predict en reproduisant le curl (headers + JSON body)."""
    payload = {"text_cleaned": text_value}
    headers = {
        "Content-Type": "application/json",
        "token": token,  # important: header "token" comme dans ton curl
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        content_type = r.headers.get("content-type", "")
        out = {
            "status_code": r.status_code,
            "content_type": content_type,
            "headers": dict(r.headers),
            "text": r.text,
        }
        # Si c'est du JSON, on tente de parser pour l'afficher proprement
        if "application/json" in content_type.lower():
            try:
                out["json"] = r.json()
            except Exception:
                out["json"] = None
        return out
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


col_ok, col_ko = st.columns(2)

with col_ok:
    if st.button("🚀 Predict (token = 456)", type="primary", use_container_width=True):
        st.session_state["last_predict"] = {
            "token": "456",
            "result": call_predict(api_url, "456", text_cleaned),
        }

with col_ko:
    if st.button("🔒 Predict (token = 100)", use_container_width=True):
        st.session_state["last_predict"] = {
            "token": "100",
            "result": call_predict(api_url, "100", text_cleaned),
        }

# Affichage du résultat si disponible
if "last_predict" in st.session_state:
    token_used = st.session_state["last_predict"]["token"]
    res = st.session_state["last_predict"]["result"]

    st.subheader("📨 Requête envoyée")

    curl_cmd = (
        f"curl -X POST {api_url} \\\n"
        f"  -H \"Content-Type: application/json\" \\\n"
        f"  -H \"token: {token_used}\" \\\n"
        f"  -d '{{\"text_cleaned\":\"{text_cleaned}\"}}'"
    )
    st.code(curl_cmd, language="bash")

    st.subheader("📬 Réponse API")

    if "error" in res:
        st.error(f"Erreur réseau / connexion : {res['error']}")
    else:
        code = res["status_code"]
        if code == 200:
            st.success(f"✅ HTTP {code} (token={token_used})")
        else:
            st.warning(f"⚠️ HTTP {code} (token={token_used})")

        # Affichage du body
        if res.get("json") is not None:
            st.json(res["json"])
        else:
            st.code(res.get("text", ""), language="text")

        with st.expander("Voir les headers de réponse"):
            st.json(res.get("headers", {}))


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Rakuten MLOps Dashboard v1.0 | Propulsé par Streamlit</p>
</div>
""", unsafe_allow_html=True)
