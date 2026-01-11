import streamlit as st
import os
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Prometheus - Rakuten MLOps",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Prometheus - Métriques Système")

st.markdown("""
Prometheus collecte et stocke les métriques de tous les services 
pour le monitoring et l'alerting.
""")

# Récupération de l'URL depuis les variables d'environnement
prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")

# Affichage de l'interface Prometheus en iframe
st.markdown("### Interface Prometheus")

# Option 1 : Iframe (si CORS autorisé)
st.markdown(f"""
<iframe src="{prometheus_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

# Option 2 : Lien direct (alternative)
st.markdown("---")
st.markdown(f"**Accès direct :** [Ouvrir Prometheus dans un nouvel onglet]({prometheus_url.replace('prometheus', 'localhost')})")

# Section requêtes PromQL
st.markdown("---")
st.markdown("### 🔎 Requêtes PromQL Rapides")

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input(
        "Entrez une requête PromQL :",
        placeholder="Ex: rate(model_requests_total[5m])"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    execute_query = st.button("🚀 Exécuter", type="primary")

# Requêtes pré-définies
st.markdown("**Requêtes pré-définies :**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Total Requêtes"):
        query = "sum(model_requests_total)"
        execute_query = True

with col2:
    if st.button("⏱️ Latence Moyenne"):
        query = "rate(model_request_duration_seconds_sum[5m]) / rate(model_request_duration_seconds_count[5m])"
        execute_query = True

with col3:
    if st.button("💾 Utilisation Mémoire"):
        query = "container_memory_usage_bytes"
        execute_query = True

# Exécution de la requête
if execute_query and query:
    try:
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['status'] == 'success' and data['data']['result']:
                st.success(f"✅ Requête exécutée avec succès - {len(data['data']['result'])} résultat(s)")
                
                # Affichage des résultats
                results = []
                for result in data['data']['result']:
                    metric_labels = result.get('metric', {})
                    value = result['value'][1]
                    
                    results.append({
                        'Métrique': metric_labels.get('__name__', 'N/A'),
                        'Labels': ', '.join([f"{k}={v}" for k, v in metric_labels.items() if k != '__name__']),
                        'Valeur': value,
                        'Timestamp': datetime.fromtimestamp(result['value'][0]).strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Graphique si plusieurs résultats
                if len(results) > 1:
                    fig = px.bar(
                        df,
                        x='Labels',
                        y='Valeur',
                        title=f'Résultats de la requête: {query}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Aucun résultat trouvé pour cette requête")
        else:
            st.error(f"❌ Erreur HTTP {response.status_code}")
    
    except Exception as e:
        st.error(f"❌ Erreur lors de l'exécution de la requête: {str(e)}")

# Informations utiles
with st.expander("ℹ️ Aide - Prometheus & PromQL"):
    st.markdown("""
    ### Métriques disponibles :
    
    **Modèle ML :**
    - `model_requests_total` : Nombre total de requêtes
    - `model_prediction_confidence` : Confiance des prédictions
    - `model_entropy_last_value` : Entropie (indicateur de drift)
    - `model_request_duration_seconds` : Latence des requêtes
    
    **Infrastructure :**
    - `container_memory_usage_bytes` : Utilisation mémoire
    - `container_cpu_usage_seconds_total` : Utilisation CPU
    - `container_network_receive_bytes_total` : Trafic réseau entrant
    
    ### Exemples de requêtes PromQL :
    
    ```promql
    # Taux de requêtes par seconde
    rate(model_requests_total[5m])
    
    # Latence moyenne
    rate(model_request_duration_seconds_sum[5m]) / 
    rate(model_request_duration_seconds_count[5m])
    
    # Top 5 des catégories prédites
    topk(5, sum by (predicted_class) (model_requests_total))
    
    # Mémoire utilisée > 500MB
    container_memory_usage_bytes > 500000000
    ```
    
    ### Opérateurs PromQL :
    
    - `rate()` : Taux de variation par seconde
    - `sum()` : Somme des valeurs
    - `avg()` : Moyenne
    - `topk()` : Top K résultats
    - `by` : Regroupement par label
    
    ### Fonctions de temps :
    
    - `[5m]` : Fenêtre de 5 minutes
    - `[1h]` : Fenêtre de 1 heure
    - `[1d]` : Fenêtre de 1 jour
    """)

# Targets Prometheus
st.markdown("---")
st.markdown("### 🎯 Targets Actives")

try:
    response = requests.get(f"{prometheus_url}/api/v1/targets", timeout=5)
    if response.status_code == 200:
        targets_data = response.json()
        active_targets = targets_data.get('data', {}).get('activeTargets', [])
        
        targets_info = []
        for target in active_targets:
            targets_info.append({
                'Service': target.get('labels', {}).get('job', 'N/A'),
                'Instance': target.get('labels', {}).get('instance', 'N/A'),
                'Statut': '✅ UP' if target.get('health') == 'up' else '❌ DOWN',
                'Dernière Collecte': target.get('lastScrape', 'N/A')
            })
        
        if targets_info:
            df_targets = pd.DataFrame(targets_info)
            st.dataframe(df_targets, use_container_width=True)
        else:
            st.info("Aucune target active trouvée")
except Exception as e:
    st.warning(f"⚠️ Impossible de récupérer les targets: {str(e)}")
