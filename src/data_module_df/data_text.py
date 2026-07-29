# data_text.py
import json
import os
import spacy

from collections import Counter
from tqdm import tqdm

# Chargement du modèle spaCy français
# Suggestion : Implémenter une fonction de lazy loading pour accélérer les tests
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    from spacy.cli import download
    print("📦 Modèle spaCy 'fr_core_news_sm' manquant. Téléchargement en cours...")
    download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")


def preprocess_text(text):
    """Nettoie et lemmatise une chaîne de texte."""
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ])


def preprocess_dataframe(df, text_col="designation_description", batch_size=1000):
    """Prétraitement par lot du texte d’un DataFrame."""
    n_cpu = os.cpu_count()
    cleaned_texts = []

    total_docs = len(df)
    print(f"🚀 Démarrage du preprocessing sur {total_docs} lignes...")
    print(f"⚙️ Pipeline actif : {[pipe for pipe in nlp.pipe_names]}")  # Vérification visuelle de ce que spacy charge

    data_stream = nlp.pipe(df[text_col].astype(str),
                           batch_size=batch_size,
                           n_process=round(n_cpu/2))

    for doc in tqdm(data_stream, total=total_docs, desc="spaCy preprocessing"):
        tokens = [
            t.lemma_.lower()
            for t in doc
            if t.is_alpha and not t.is_stop
        ]
        cleaned_texts.append(" ".join(tokens))

    df["text_cleaned"] = cleaned_texts
    return df


def save_class_distribution(df, label_col, output_path):
    """Sauvegarde la répartition des classes sous forme JSON."""
    class_counts = dict(Counter(df[label_col]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(class_counts, f, indent=2, ensure_ascii=False)
    print(f"📊 Répartition des classes sauvegardée dans {output_path}")
