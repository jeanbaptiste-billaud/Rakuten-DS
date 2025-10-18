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
