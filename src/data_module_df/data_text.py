# data_text.py
import langid
import pandas as pd
import swifter
import spacy
from deep_translator import GoogleTranslator
from spacy.cli import download
from tqdm import tqdm

# Chargement du modèle spaCy français
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("📦 Modèle spaCy 'fr_core_news_sm' manquant. Téléchargement en cours...")
    download("fr_core_news_sm")


def truncate_text(text, max_length=2000, threshold=5000):
    if isinstance(text, str) and len(text) > threshold:
        return text[:max_length]
    return text


def detect_language(text):
    try:
        return langid.classify(text)[0]
    except:
        return "unknown"


def translate_texts_to_french(texts, batch_size=50):
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            translated.extend(
                GoogleTranslator(source="auto", target='fr').translate_batch(batch)
            )
        except Exception as e:
            print(f"⚠️ Erreur batch {i}-{i+batch_size} : {e}")
            translated.extend(batch)  # fallback : texte inchangé
    return translated


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

    # Étape 1 : troncature
    # data.loc[:, input_col] = data.loc[:, input_col].swifter.apply(truncate_text)

    # # Étape 2 : détection et traduction
    # data["language"] = data.loc[:, input_col].swifter.apply(detect_language)
    #
    # # Étape 3 : initialisation de la colonne traduite
    # data["text_fr"] = data[input_col]
    #
    # # Étape 4 : traitement par groupe de langue ≠ 'fr'
    # print("\n🌍 Traduction vers le français par langue :")
    # for lang, group_df in tqdm(data.groupby("language"), desc="🗣️ Langues détectées"):
    #     if lang == "fr":
    #         continue
    #
    #     to_translate = group_df[input_col].fillna("").tolist()
    #     translated = translate_texts_to_french(
    #         texts=to_translate,
    #         batch_size=cfg.training.batch_size
    #     )
    #
    #     data.loc[group_df.index, "text_fr"] = translated

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