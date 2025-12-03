# data_balancing.py
import pandas as pd
import numpy as np


def generate_text_dataset(df, target_size=None, min_per_class=300, random_state=42, save_path=None):
    """
    Génère un dataset textuel partiellement équilibré à partir du CSV brut.

    Étapes :
      1️⃣ Tire min_per_class échantillons par classe (équilibrage minimal).
      2️⃣ Retire ces échantillons du pool original.
      3️⃣ Tire aléatoirement les échantillons restants pour atteindre target_size.

    Args:
        df (pd.DataFrame): DataFrame contenant au moins ['designation', 'description', 'prdtypecode']
        target_size (int or None): Nombre total d'échantillons souhaité (None = tout le dataset)
        min_per_class (int): Minimum d'échantillons garantis par classe
        random_state (int): Graine aléatoire
        save_path (str): Chemin CSV de sortie (même nom pour DVC tracking)

    Returns:
        pd.DataFrame: Dataset partiellement équilibré
    """
    np.random.seed(random_state)
    df = df.copy()
    df['text'] = df['designation'].fillna('') + ' ' + df['description'].fillna('')

    # Comptage initial
    class_counts = df['prdtypecode'].value_counts().sort_index()
    print(f"Nombre total de classes : {len(class_counts)}")

    # Étape 1 : échantillons équilibrés minimaux
    balanced_parts = []
    used_indices = set()

    for cls, count in class_counts.items():
        subset = df[df['prdtypecode'] == cls]
        n_samples = min(min_per_class, len(subset))
        sampled = subset.sample(n=n_samples, random_state=random_state)
        balanced_parts.append(sampled)
        used_indices.update(sampled.index)

    base_df = pd.concat(balanced_parts)
    print(f"⚖️ {len(base_df)} échantillons après équilibrage minimal ({min_per_class}/classe)")

    # Étape 2 : tirage aléatoire global pour compléter
    if target_size is not None and len(base_df) < target_size:
        remaining_needed = target_size - len(base_df)
        remaining_df = df.loc[~df.index.isin(used_indices)]

        if len(remaining_df) < remaining_needed:
            print(f"⚠️ Seulement {len(remaining_df)} échantillons restants, inférieur à {remaining_needed}")
            remaining_needed = len(remaining_df)

        extra_samples = remaining_df.sample(n=remaining_needed, random_state=random_state)
        final_df = pd.concat([base_df, extra_samples])
    else:
        final_df = base_df

    final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"✅ Dataset final : {len(final_df)} échantillons")
    print("📊 Distribution (aperçu) :")
    print(final_df['prdtypecode'].value_counts().sort_index().head(10))

    if save_path:
        final_df.to_csv(save_path, index=False)
        print(f"💾 Dataset sauvegardé dans {save_path}")

    return final_df


class Categories:
    def __init__(self):
        """Initialise le mapping des catégories Rakuten"""

        # Mapping officiel prdtypecode → nom de catégorie
        self.category_names = {
            10: "Livres",
            2280: "Jeux vidéo",
            50: "Jouets & Jeux",
            1280: "Accessoires téléphones",
            2705: "Accessoires console",
            2522: "Équipement bébé",
            2582: "Matériel & accessoires",
            1560: "Photos",
            1281: "Téléphonie fixe",
            1920: "Musique amplifiée",
            2403: "Livres en langues étrangères",
            1140: "TV",
            2583: "Articles sport",
            1180: "Décoration",
            1300: "Jeux vidéo ancien",
            2462: "Fournitures bureau",
            1160: "Électroménager",
            2060: "Articles soins",
            40: "DVD & Films",
            60: "Consoles",
            1320: "CD",
            1302: "Jeux vidéo rétro",
            2220: "Puériculture",
            2905: "Instruments musique",
            2585: "Sports & Loisirs",
            1940: "Instrument musique",
            1301: "Consoles rétro",
        }

        # Mapping prdtypecode → index interne consécutif (0, 1, 2, …)
        self.category_to_idx = {
            code: idx for idx, code in enumerate(sorted(self.category_names.keys()))
        }

        # Mapping inverse index → prdtypecode
        self.idx_to_category = {
            idx: code for code, idx in self.category_to_idx.items()
        }
