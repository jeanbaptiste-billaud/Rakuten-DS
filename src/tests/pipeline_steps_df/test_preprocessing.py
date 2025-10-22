import pandas as pd
import pandas.testing as pd_testing
import pytest
import numpy as np

from src.pipeline_steps_df import preprocessing

# --- Fixtures : Données factices ---

@pytest.fixture
def fake_raw_df():
    """DataFrame factice retourné par pd.read_csv."""
    return pd.DataFrame({
        "designation": ["Livre A", np.nan, "Jeu B"],
        "description": ["Description A", "Description B", np.nan],
        "prdtypecode": [10, 20, 10]
    })

@pytest.fixture
def fake_preprocessed_df(fake_raw_df): # Basé sur le fake_raw_df
    """
    DataFrame factice retourné par la fonction (mockée) preprocess_dataframe.
    C'est ici que nous appliquons la modification.
    """
    # Recréer l'état que le script 'main' attend *après* la fusion 
    # mais *avant* l'appel à preprocess_dataframe
    df = fake_raw_df.copy()
    df["designation_description"] = ["Livre A Description A", "Description B", "Jeu B"]
    
    # Simuler le retour de la fonction preprocess_dataframe
    df["text_cleaned"] = ["livre a description a", "description b", "jeu b"]
 
    
    return df

# --- Test principal (avec Mocks) ---

def test_preprocessing_main_orchestration(mocker, fake_raw_df, fake_preprocessed_df):
    """
    Teste le script main() de preprocessing en simulant 
    toutes les lectures/écritures et la logique métier importée.
    """
    
    # 1. CONFIGURATION DES MOCKS
    mock_get_root = mocker.patch("src.pipeline_steps_df.preprocessing.get_project_root")
    mock_get_root.return_value = "/fake/root"
    
    mock_makedirs = mocker.patch("os.makedirs")
    
    mock_read_csv = mocker.patch("pandas.read_csv")
    mock_read_csv.return_value = fake_raw_df.copy()
    
    # Simuler la logique métier importée
    mock_preprocess_df = mocker.patch("src.pipeline_steps_df.preprocessing.preprocess_dataframe")
    # Elle retourne notre fixture corrigée
    mock_preprocess_df.return_value = fake_preprocessed_df 
    
    mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
    mock_save_distrib = mocker.patch("src.pipeline_steps_df.preprocessing.save_class_distribution")

    # Définir les chemins attendus
    expected_raw_path = "/fake/root/data/dataset/raw_dataset.csv"
    expected_output_dir = "/fake/root/data/preprocessed"
    expected_processed_path = "/fake/root/data/preprocessed/preprocessed_text.csv"
    expected_distrib_path = "/fake/root/data/preprocessed/class_distribution.json"

    # 2. EXÉCUTION
    preprocessing.main()

    # 3. VÉRIFICATION (Asserts)
    
    # A-t-on lu le bon fichier ?
    mock_read_csv.assert_called_once_with(expected_raw_path)
    
    # A-t-on appelé 'preprocess_dataframe' avec le bon DataFrame ?
    call_args = mock_preprocess_df.call_args[0]
    df_passed_to_preprocess = call_args[0]
    
    # Vérifie que la colonne 'designation_description' a été créée
    assert 'designation_description' in df_passed_to_preprocess.columns
    
    # A-t-on sauvegardé le bon DataFrame (celui retourné par le mock) ?
    # 'mock_to_csv' est une méthode de l'objet fake_preprocessed_df
    mock_to_csv.assert_called_once_with(expected_processed_path, index=False)
    
    # A-t-on sauvegardé la distribution de la classe ?
    mock_save_distrib.assert_called_once_with(
        fake_preprocessed_df,       # Le DataFrame final
        "prdtypecode",              # La colonne de classe
        expected_distrib_path       # Le chemin de sortie
    )