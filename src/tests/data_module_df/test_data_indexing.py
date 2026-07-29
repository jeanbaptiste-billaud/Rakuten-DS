import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import os

# Importer les fonctions à tester
from src.data_module_df.data_indexing import (
    generate_splits,
    save_indices,
    load_indices
)

# --- Fixture pour les données ---

@pytest.fixture
def sample_y():
    """
    Crée un 'y' (labels) de 1000 échantillons avec 3 classes
    très déséquilibrées pour tester la stratification.
    - Classe 10: 800 (80%)
    - Classe 20: 150 (15%)
    - Classe 30: 50 (5%)
    """
    y = np.concatenate([
        np.repeat(10, 800),
        np.repeat(20, 150),
        np.repeat(30, 50)
    ])
    # Mélanger pour simuler un vrai dataset
    rng = np.random.default_rng(42)
    rng.shuffle(y)
    return y

# --- Tests pour generate_splits ---

def test_generate_splits_disjoint_and_complete(sample_y):
    """
    Test 1: Vérifie que les 3 ensembles d'indices (train, val, test)
    n'ont aucun élément en commun (disjoints) et que leur union
    reconstitue l'ensemble des indices d'origine (complet).
    """
    y = sample_y
    indices = generate_splits(y, test_size=0.2, val_size=0.2, random_state=42)
    
    train_idx = set(indices['train_idx'])
    val_idx = set(indices['val_idx'])
    test_idx = set(indices['test_idx'])
    
    # 1. Vérifier qu'ils sont disjoints
    assert train_idx.isdisjoint(val_idx), "Train et Val se chevauchent"
    assert train_idx.isdisjoint(test_idx), "Train et Test se chevauchent"
    assert val_idx.isdisjoint(test_idx), "Val et Test se chevauchent"
    
    # 2. Vérifier qu'ils sont complets
    full_set = train_idx.union(val_idx).union(test_idx)
    original_set = set(range(len(y)))
    
    assert full_set == original_set, "L'union des splits ne couvre pas tout le dataset"

def test_generate_splits_sizes(sample_y):
    """
    Test 2: Vérifie que les tailles des splits sont correctes.
    - Total = 1000
    - test_size = 0.2 -> test = 1000 * 0.2 = 200
    - train_val = 1000 - 200 = 800
    - val_size = 0.2 -> val = 800 * 0.2 = 160
    - train = 800 - 160 = 640
    """
    y = sample_y
    indices = generate_splits(y, test_size=0.2, val_size=0.2, random_state=42)
    
    assert len(indices['test_idx']) == 200
    assert len(indices['val_idx']) == 160
    assert len(indices['train_idx']) == 640

def test_generate_splits_stratification(sample_y):
    """
    Test 3: Vérifie que la proportion des classes est maintenue
    dans chaque split. C'est le test le plus important.
    """
    y = sample_y
    indices = generate_splits(y, test_size=0.2, val_size=0.2, random_state=42)
    
    # Proportions d'origine (10: 0.8, 20: 0.15, 30: 0.05)
    original_props = pd.Series(y).value_counts(normalize=True).sort_index()
    
    # Proportions dans les splits
    y_train = y[indices['train_idx']]
    y_val = y[indices['val_idx']]
    y_test = y[indices['test_idx']]
    
    train_props = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_props = pd.Series(y_val).value_counts(normalize=True).sort_index()
    test_props = pd.Series(y_test).value_counts(normalize=True).sort_index()
    
    # Vérifie que les proportions sont très proches (atol=0.01 -> 1% de tolérance)
    npt.assert_allclose(original_props.values, train_props.values, atol=0.01)
    npt.assert_allclose(original_props.values, val_props.values, atol=0.01)
    npt.assert_allclose(original_props.values, test_props.values, atol=0.01)

def test_generate_splits_reproducibility(sample_y):
    """
    Test 4: Vérifie que deux appels avec le même random_state
    produisent exactement les mêmes indices.
    """
    y = sample_y
    
    indices_run1 = generate_splits(y, random_state=42)
    indices_run2 = generate_splits(y, random_state=42)
    
    npt.assert_array_equal(indices_run1['train_idx'], indices_run2['train_idx'])
    npt.assert_array_equal(indices_run1['val_idx'], indices_run2['val_idx'])
    npt.assert_array_equal(indices_run1['test_idx'], indices_run2['test_idx'])

# --- Tests pour save_indices et load_indices ---

def test_save_and_load_indices_roundtrip(tmp_path):
    """
    Test 5: Fait un "aller-retour".
    1. Crée un dictionnaire factice.
    2. Le sauvegarde dans un dossier temporaire (tmp_path).
    3. Vérifie que le fichier (et le dossier) a été créé.
    4. Le recharge.
    5. Vérifie que le dictionnaire rechargé est identique à l'original.
    """
    # tmp_path est une fixture pytest qui fournit un dossier temporaire
    
    # 1. Dictionnaire factice
    dict_to_save = {
        'train': np.array([1, 5, 9]),
        'val': np.array([2, 6]),
        'test': np.array([3, 4, 7, 8])
    }
    
    # 2. Sauvegarde
    # On crée un sous-dossier pour vérifier que os.makedirs fonctionne
    save_dir = tmp_path / "indices_output"
    file_path = save_dir / "my_indices.npz"
    
    assert not save_dir.exists() # Le dossier n'existe pas encore
    
    save_indices(dict_to_save, file_path)
    
    # 3. Vérifier la création
    assert save_dir.exists(), "Le dossier aurait dû être créé"
    assert file_path.exists(), "Le fichier .npz n'a pas été créé"
    
    # 4. Rechargement
    loaded_dict = load_indices(file_path)
    
    # 5. Vérification
    assert isinstance(loaded_dict, dict), "load_indices doit retourner un dict"
    assert loaded_dict.keys() == dict_to_save.keys()
    
    # Comparaison des arrays numpy
    npt.assert_array_equal(loaded_dict['train'], dict_to_save['train'])
    npt.assert_array_equal(loaded_dict['val'], dict_to_save['val'])
    npt.assert_array_equal(loaded_dict['test'], dict_to_save['test'])