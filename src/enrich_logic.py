import pandas as pd
import numpy as np

def select_new_samples(df_raw: pd.DataFrame, 
                       df_all: pd.DataFrame, 
                       n_new_samples: int, 
                       seed: int) -> pd.DataFrame:
    """
    Identifie et échantillonne de nouveaux échantillons depuis df_all 
    qui ne sont pas dans df_raw, basé sur 'productid'.
    """
    np.random.seed(seed)

    # Identifier les nouveaux samples
    used_ids = set(df_raw['productid'])
    df_candidates = df_all[~df_all['productid'].isin(used_ids)]

    # Valider la disponibilité
    if len(df_candidates) < n_new_samples:
        raise ValueError(
            f"Pas assez de nouveaux échantillons disponibles "
            f"({len(df_candidates)} restants, {n_new_samples} demandés)."
        )
    
    # Échantillonner
    df_new = df_candidates.sample(n=n_new_samples, random_state=seed)
    return df_new

def enrich_dataset(df_raw: pd.DataFrame, 
                   df_new: pd.DataFrame) -> pd.DataFrame:
    """Concatène l'ancien et le nouveau dataset."""
    return pd.concat([df_raw, df_new], ignore_index=True)