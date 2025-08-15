# data_balancing.py
import os
import numpy as np
import pandas as pd
from collections import Counter
import logging


def undersample(X, y, max_per_class=1000, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame({'X': list(X), 'y': y})
    sampled_dfs = []

    for label in np.unique(y):
        df_class = df[df['y'] == label]
        if len(df_class) > max_per_class:
            sampled = df_class.sample(n=max_per_class, random_state=random_state)
        else:
            sampled = df_class
        sampled_dfs.append(sampled)

    balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return np.array(balanced_df['X'].tolist()), balanced_df['y'].values


def print_class_distribution(y, title="Répartition des classes"):
    counter = Counter(y)
    print(f"\n📊 {title}:")
    for label, count in counter.items():
        print(f"  Classe {label}: {count} échantillons")


def create_balanced_dataset(config, raw_csv_df):
    """
    Crée un dataset équilibré en considérant à la fois la distribution des classes
    et la taille des fichiers images.

    Args:
        raw_csv_df (pd.DataFrame): DataFrame contenant les métadonnées des images et leurs labels

    Returns:
        List[int]: Liste des indices sélectionnés pour le dataset équilibré
    """
    file_info = []

    # Collecte des informations sur les fichiers
    for _, row in raw_csv_df.iterrows():
        image_file = f"image_{row['imageid']}_product_{row['productid']}.jpg"
        file_path = os.path.join(config.image_dir, image_file)

        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            file_info.append({
                'index': row.name,
                'size_kb': size_kb,
                'prdtypecode': row['prdtypecode'],
                'imageid': row['imageid'],
                'productid': row['productid']
            })

    df_analysis = pd.DataFrame(file_info)
    df_analysis.set_index('index', inplace=True)

    balanced_indices = []
    # category_names = Categories().category_names

    # Pour chaque classe
    for classe in df_analysis['prdtypecode'].unique():
        class_data = df_analysis[df_analysis['prdtypecode'] == classe].copy()
        n_samples = len(class_data)

        if n_samples > config.training.target_size:
            # Sous-échantillonnage stratifié par taille
            size_bins = pd.qcut(class_data['size_kb'], q=5, labels=False)
            class_data['size_bin'] = size_bins
            samples_per_bin = config.training.target_size // 5

            stratified_sample = []
            for bin_id in range(5):
                bin_data = class_data[class_data['size_bin'] == bin_id]
                if len(bin_data) > 0:
                    selected = bin_data.sample(
                        n=min(len(bin_data), samples_per_bin),
                        random_state=config.training.random_state
                    ).index.tolist()
                    stratified_sample.extend(selected)

            # Si on n'a pas assez d'échantillons après stratification
            remaining = config.training.target_size - len(stratified_sample)
            if remaining > 0:
                additional = class_data[~class_data.index.isin(stratified_sample)].sample(
                    n=min(remaining, len(class_data) - len(stratified_sample)),
                    random_state=config.training.random_state
                ).index.tolist()
                stratified_sample.extend(additional)

            balanced_indices.extend(stratified_sample)

        else:
            # Sur-échantillonnage stratifié par taille
            current_indices = class_data.index.tolist()
            balanced_indices.extend(current_indices)  # Ajoute d'abord tous les échantillons existants

            if n_samples > 0:
                # Calcul du nombre d'échantillons supplémentaires nécessaires
                n_needed = config.training.target_size - n_samples

                # Division en bins de taille
                size_bins = pd.qcut(class_data['size_kb'], q=min(5, n_samples), labels=False)
                class_data['size_bin'] = size_bins

                # Sur-échantillonnage par bin
                additional_samples = []
                samples_needed_per_bin = n_needed // len(class_data['size_bin'].unique())

                for bin_id in class_data['size_bin'].unique():
                    bin_data = class_data[class_data['size_bin'] == bin_id]
                    if len(bin_data) > 0:
                        bin_indices = bin_data.index.tolist()
                        additional = np.random.choice(
                            bin_indices,
                            size=samples_needed_per_bin,
                            replace=True
                        ).tolist()
                        additional_samples.extend(additional)

                # Gestion du reste
                remaining = n_needed - len(additional_samples)
                if remaining > 0:
                    extra = np.random.choice(
                        current_indices,
                        size=remaining,
                        replace=True
                    ).tolist()
                    additional_samples.extend(extra)

                balanced_indices.extend(additional_samples)

    # Vérification finale
    # logger.info(f"Indices retenus: {len(balanced_indices)} sur {len(df_analysis)} images")
    # for classe in df_analysis['prdtypecode'].unique():
    #     n_class = sum(df_analysis.loc[balanced_indices, 'prdtypecode'] == classe)
    #     logger.info(f"Classe {classe} ({category_names[classe]}): {n_class} images")

    return balanced_indices


class Categories:
    def __init__(self):
        """Initialise le mapping des catégories Rakuten"""
        self.category_names = {
            10: "Livres", 2280: "Jeux vidéo", 50: "Jouets & Jeux",
            1280: "Accessoires téléphones", 2705: "Accessoires console",
            2522: "Équipement bébé", 2582: "Matériel & accessoires",
            1560: "Photos", 1281: "Téléphonie fixe",
            1920: "Musique amplifiée", 2403: "Livres en langues étrangères",
            1140: "TV", 2583: "Articles sport", 1180: "Décoration",
            1300: "Jeux vidéo ancien", 2462: "Fournitures bureau",
            1160: "Électroménager", 2060: "Articles soins",
            40: "DVD & Films", 60: "Consoles", 1320: "CD",
            1302: "Jeux vidéo rétro", 2220: "Puériculture",
            2905: "Instruments musique", 2585: "Sports & Loisirs",
            1940: "Instrument musique", 1301: "Consoles rétro"
        }
        # Créer le mapping vers des indices consécutifs
        self.category_to_idx = {code: idx for idx, code in enumerate(sorted(self.category_names.keys()))}
        self.idx_to_category = {idx: code for code, idx in self.category_to_idx.items()}
