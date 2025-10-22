import pytest
import pandas as pd
import pandas.testing as pd_testing
from src.pipeline_steps_df.enrich_logic import select_new_samples, enrich_dataset

# --- Fixtures : fausses données pour nos tests ---

@pytest.fixture
def sample_df_raw() -> pd.DataFrame:
    """Le dataset qui existe déjà."""
    return pd.DataFrame({
        "productid": [104324, 205435],
        "designation": ["Outils de jardin", "Jouet pour enfant"],
        "description": ["Set de jardinage complet avec outils ergonomique", ""],
        "imageid": [54839, 25392],
        "prdtypecode": [1478, 2405]
    })

@pytest.fixture
def sample_df_all() -> pd.DataFrame:
    """Toutes les données disponibles."""
    return pd.DataFrame({
    "productid": [104324, 205435, 308976, 401234, 507890],
    "designation": [
        "Outils de jardin", 
        "Jouet pour enfant", 
        "Livre de cuisine", 
        "Casque audio Bluetooth", 
        "Tapis de yoga"
    ],
    "description": [
        "Set de jardinage complet avec outils ergonomiques", 
        "Puzzle 1000 pièces, thème animaux de la forêt", 
        "Recettes faciles pour étudiants par un chef étoilé", 
        "Casque sans fil à réduction de bruit active, autonomie 30h", 
        "Tapis en caoutchouc naturel, antidérapant et écologique"
    ],
    "imageid": [54839, 25392, 67890, 12345, 98765],
    "prdtypecode": [1478, 2405, 1010, 2020, 1850]
})

# --- Tests pour select_new_samples ---

def test_select_new_samples_nominal(sample_df_raw, sample_df_all):
    """
    Test 1 (Cas nominal) : Vérifie que la fonction sélectionne 
    correctement 2 nouveaux échantillons.
    """
    n_new_samples = 2
    seed = 42
    
    df_new = select_new_samples(sample_df_raw, sample_df_all, n_new_samples, seed)
    
    # 1. Vérifie la taille
    assert len(df_new) == n_new_samples
    
    # 2. Vérifie que les IDs sont bien nouveaux (pas 10 ou 20)
    existing_ids = set(sample_df_raw['productid'])
    new_ids = set(df_new['productid'])
    assert existing_ids.isdisjoint(new_ids) # Vérifie qu'il n'y a pas d'intersection
    
    # 3. Vérifie que les IDs proviennent bien de df_all
    all_ids = set(sample_df_all['productid'])
    assert new_ids.issubset(all_ids)

def test_select_new_samples_raises_error_if_not_enough(sample_df_raw, sample_df_all):
    """
    Test 2 (Cas d'erreur) : Vérifie qu'une erreur est levée si 
    on demande plus d'échantillons qu'il n'y en a de disponibles.
    """
    # Il n'y a que 3 nouveaux samples (30, 40, 50), on en demande 5.
    n_new_samples = 5
    
    with pytest.raises(ValueError, match="Pas assez de nouveaux échantillons"):
        select_new_samples(sample_df_raw, sample_df_all, n_new_samples, seed=42)

def test_select_new_samples_is_reproducible(sample_df_raw, sample_df_all):
    """
    Test 3 (Reproductibilité) : Vérifie que le même 'seed' 
    donne deux fois le même résultat.
    """
    df_new_run1 = select_new_samples(sample_df_raw, sample_df_all, n_new_samples=2, seed=42)
    df_new_run2 = select_new_samples(sample_df_raw, sample_df_all, n_new_samples=2, seed=42)
    
    # Utilise l'utilitaire de test de pandas pour comparer les DataFrames
    pd_testing.assert_frame_equal(df_new_run1, df_new_run2)

def test_select_new_samples_no_new_samples_available(sample_df_all):
    """
    Test 4 (Cas limite) : Vérifie que le test échoue si df_raw 
    contient déjà tous les samples.
    """
    # df_raw est identique à df_all
    df_raw_identical = sample_df_all.copy()
    
    with pytest.raises(ValueError):
        select_new_samples(df_raw_identical, sample_df_all, n_new_samples=1, seed=42)

# --- Test pour enrich_dataset ---

def test_enrich_dataset(sample_df_raw):
    """Test 5 (Concaténation) : Vérifie la simple concaténation."""
    df_new = pd.DataFrame({"productid": [30], "feature_a": [3]})
    
    df_updated = enrich_dataset(sample_df_raw, df_new)
    
    # Le nouveau DF doit avoir la somme des lignes
    assert len(df_updated) == len(sample_df_raw) + len(df_new)
    # Vérifie que le dernier ID est bien celui ajouté
    assert df_updated.iloc[-1]['productid'] == 30