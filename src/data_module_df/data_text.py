# data_text.py
import spacy
from spacy.cli import download

# Chargement du modèle spaCy français
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("📦 Modèle spaCy 'fr_core_news_sm' manquant. Téléchargement en cours...")
    download("fr_core_news_sm")


def preprocess_text_fr(text):
    if not isinstance(text, str):
        return ""
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ])


def batch_preprocess_text_fr(texts, batch_size=50, n_process=-1):
    """
    Traite une liste de textes avec spaCy en mode pipeline optimisé.
    Nettoyage = lemmatisation + suppression des stopwords/ponctuation.
    """
    flux = zip(texts.index, df["text"].astype(str))
    cleaned = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        cleaned.append(" ".join(tokens))
    return cleaned


def prepare_text_column(df, cfg, input_col="designation_description"):
    data = df.copy()

    # Étape 5 : NLP avec spaCy français
    cleaned = []
    flux = zip(data[input_col].astype(str), data.index)
    for doc, idx in nlp.pipe(flux, as_tuples=True, n_process=cfg.num_workers, batch_size=cfg.training.batch_size):
        tokens = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
        cleaned.append((idx, tokens))

    data["text_fr_cleaned"] = [t for _, t in sorted(cleaned, key=lambda x: x[0])]

    # data["text_fr_cleaned"] = batch_preprocess_text_fr(
    #     data[input_col].fillna(""),
    #     batch_size=cfg.training.batch_size,
    #     n_process=cfg.num_workers
    # )

    # return data.drop(columns=["language", "text_fr"])
    return data
