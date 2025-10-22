import pytest
import pandas as pd
import pandas.testing as pd_testing
import numpy as np

# Importer la fonction à tester
from src.data_module_df.data_balancing import generate_text_dataset

# --- Fixture : Données de test ---

@pytest.fixture
def sample_df():
    """
    Un DataFrame de test avec une distribution de classes inégale.
    - Classe 10: 10 échantillons
    - Classe 20: 5 échantillons
    - Classe 30: 2 échantillons (!! Moins que min_per_class habituel)
    Total : 17 échantillons
    """
    data = {
        # 'designation' et 'description' pour tester la fusion de 'text'
        "designation": [f"Design {i}" for i in range(17)],
        "description": [f"Desc {i}" for i in range(17)],
        "prdtypecode": [10]*10 + [20]*5 + [30]*2,
        # 'productid' pour suivre les échantillons
        "productid": range(1, 18) 
    }
    # Introduire des NaN pour tester la robustesse
    data["designation"][5] = np.nan  # Ligne avec productid 6
    data["description"][8] = np.nan  # Ligne avec productid 9
    
    return pd.DataFrame(data)

# --- Tests de Logique ---

def test_nominal_case_balancing(sample_df):
    """
    Test 1: Cas nominal.
    - min_per_class = 3
    - target_size = 10
    
    Attendu :
    - Base équilibrée = 3 (de 10) + 3 (de 20) + 2 (de 30) = 8 échantillons
    - Complément = 2 échantillons aléatoires (pour atteindre 10)
    - Total = 10 échantillons
    - Classe 10 doit avoir >= 3
    - Classe 20 doit avoir >= 3
    - Classe 30 doit avoir == 2
    """
    min_c = 3
    target = 10
    
    final_df = generate_text_dataset(
        sample_df, 
        target_size=target, 
        min_per_class=min_c, 
        random_state=42
    )
    
    # 1. Vérifier la taille finale
    assert len(final_df) == target
    
    # 2. Vérifier la distribution
    class_counts = final_df['prdtypecode'].value_counts()
    
    # Les classes 10 et 20 doivent avoir AU MOINS min_per_class
    assert class_counts[10] >= min_c
    assert class_counts[20] >= min_c
    
    # La classe 30 (qui n'a que 2 échantillons) doit être entièrement incluse
    assert class_counts[30] == 2
    

def test_handles_small_classes(sample_df):
    """
    Test 2: S'assurer que les classes avec moins de 'min_per_class' 
    sont entièrement incluses.
    """
    # On demande 5 par classe, mais la classe 30 n'en a que 2
    min_c = 5
    
    final_df = generate_text_dataset(
        sample_df, 
        target_size=None, # Ne pas compléter, juste la base
        min_per_class=min_c, 
        random_state=42
    )
    
    # Taille attendue = 5 (de 10) + 5 (de 20) + 2 (de 30) = 12
    assert len(final_df) == 12
    
    class_counts = final_df['prdtypecode'].value_counts()
    assert class_counts[10] == 5
    assert class_counts[20] == 5
    assert class_counts[30] == 2 # L'assertion clé

def test_target_size_smaller_than_balanced_base(sample_df):
    """
    Test 3: Si target_size est plus PETIT que la base équilibrée, 
    la fonction doit retourner la base équilibrée.
    """
    min_c = 3
    # Base équilibrée = 3 (10) + 3 (20) + 2 (30) = 8 échantillons
    target = 5 # Cible < Base
    
    final_df = generate_text_dataset(
        sample_df, 
        target_size=target, 
        min_per_class=min_c, 
        random_state=42
    )
    
    # Le résultat doit être la base (8), pas la cible (5)
    assert len(final_df) == 8

def test_target_size_larger_than_total(sample_df):
    """
    Test 4: Si target_size est plus GRAND que le dataset total,
    la fonction doit retourner le dataset total (mélangé).
    """
    target = 100 # Le DF n'a que 17 échantillons
    
    final_df = generate_text_dataset(
        sample_df, 
        target_size=target, 
        min_per_class=3, 
        random_state=42
    )
    
    # Le résultat doit être plafonné à la taille du DF original
    assert len(final_df) == len(sample_df)
    
    # Doit contenir tous les productid originaux
    assert set(final_df['productid']) == set(sample_df['productid'])

def test_reproducibility(sample_df):
    """
    Test 5: Vérifier que random_state garantit un résultat identique.
    """
    df_run1 = generate_text_dataset(
        sample_df, target_size=10, min_per_class=3, random_state=42
    )
    df_run2 = generate_text_dataset(
        sample_df, target_size=10, min_per_class=3, random_state=42
    )
    
    # Utilitaire de test Pandas pour comparer les DataFrames entiers
    pd_testing.assert_frame_equal(df_run1, df_run2)

# --- Tests de Fonctionnalité et Effets de Bord ---

def test_text_column_creation_with_nans(sample_df):
    """
    Test 6: Vérifier la création de la colonne 'text' et la gestion des NaN.
    """
    final_df = generate_text_dataset(
        sample_df, target_size=17, min_per_class=1, random_state=42
    )
    
    assert 'text' in final_df.columns
    
    # Cas normal (productid 1)
    text_1 = final_df[final_df['productid'] == 1]['text'].iloc[0]
    assert text_1 == "Design 0 Desc 0"
    
    # Cas NaN dans 'designation' (productid 6)
    text_6 = final_df[final_df['productid'] == 6]['text'].iloc[0]
    assert text_6 == " Desc 5" # Note l'espace
    
    # Cas NaN dans 'description' (productid 9)
    text_9 = final_df[final_df['productid'] == 9]['text'].iloc[0]
    assert text_9 == "Design 8 " # Note l'espace

def test_save_path_calls_to_csv(sample_df, mocker):
    """
    Test 7: Vérifier que la fonction to_csv est appelée (avec mock).
    """
    # Créer un "espion" pour la méthode to_csv de DataFrame
    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
    fake_path = "data/fake/output.csv"
    
    # Exécuter la fonction AVEC un save_path
    generate_text_dataset(
        sample_df, 
        target_size=10, 
        min_per_class=3, 
        random_state=42, 
        save_path=fake_path
    )
    
    # Vérifier que to_csv a été appelée une fois, avec les bons arguments
    mock_to_csv.assert_called_once_with(fake_path, index=False)
    
    # ---
    
    # Réinitialiser l'espion
    mock_to_csv.reset_mock()
    
    # Exécuter la fonction SANS save_path
    generate_text_dataset(
        sample_df, 
        target_size=10, 
        min_per_class=3, 
        random_state=42, 
        save_path=None
    )
    
    # Vérifier que to_csv n'a PAS été appelée
    mock_to_csv.assert_not_called()