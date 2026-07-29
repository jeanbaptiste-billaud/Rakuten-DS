import pandas as pd
import pytest

# On importe le script que l'on veut tester
from src.pipeline_steps_df.data import make_original_dataset


# --- Fixtures : Données factices ---

@pytest.fixture
def fake_raw_df():
    """DataFrame factice que pd.read_csv est censé retourner."""
    # Le 'prdtypecode' est crucial pour la fonction de balancing
    return pd.DataFrame({
        "productid": [1, 2, 3, 4, 5],
        "designation": ["A", "B", "C", "D", "E"],
        "prdtypecode": [10, 10, 20, 20, 30] 
    })

@pytest.fixture
def fake_balanced_df():
    """DataFrame factice que generate_text_dataset est censé retourner."""
    return pd.DataFrame({
        "productid": [1, 3, 5],
        "designation": ["A", "C", "E"],
        "prdtypecode": [10, 20, 30]
    })

# --- Test principal (avec Mocks) ---

def test_main_orchestration(mocker, fake_raw_df, fake_balanced_df):
    """
    Teste le script main() en "mockant" (simulant) toutes 
    ses dépendances externes (lecture de fichiers et logique de balancing).
    """
    
    # 1. CONFIGURATION DES MOCKS
    
    # Mock de get_project_root()
    # On lui dit de retourner un faux chemin
    mock_get_root = mocker.patch("src.pipeline_steps_df.make_original_dataset.get_project_root")
    mock_get_root.return_value = "/fake/project/root"
    
    # Mock de pd.read_csv()
    # On lui dit de retourner notre DataFrame factice
    mock_read_csv = mocker.patch("pandas.read_csv")
    mock_read_csv.return_value = fake_raw_df
    
    # Mock de la fonction de logique importée
    # On lui dit de retourner notre DataFrame équilibré factice
    mock_generate_text = mocker.patch("src.pipeline_steps_df.make_original_dataset.generate_text_dataset")
    mock_generate_text.return_value = fake_balanced_df
    
    # Mock des fonctions de logging pour vérifier les messages
    mock_log_info = mocker.patch("logging.info")
    
    # Définir les chemins attendus basés sur le faux chemin racine
    expected_input_csv = "/fake/project/root/data/raw/all_raw_data.csv"
    expected_output_csv = "/fake/project/root/data/dataset/raw_dataset.csv"

    # 2. EXÉCUTION
    # On lance la fonction main() du script
    make_original_dataset.main()

    # 3. VÉRIFICATION (Asserts)
    
    # A-t-on bien appelé get_project_root ?
    mock_get_root.assert_called_once()
    
    # A-t-on bien lu le bon fichier CSV ?
    mock_read_csv.assert_called_once_with(expected_input_csv)
    
    # A-t-on bien appelé la logique de balancing AVEC LES BONS ARGUMENTS ?
    mock_generate_text.assert_called_once_with(
        fake_raw_df,  # Le DataFrame lu par pd.read_csv
        target_size=10_000,
        min_per_class=300,
        random_state=42,
        save_path=expected_output_csv # Le chemin de sauvegarde construit
    )
    
    # A-t-on bien loggé les bonnes informations ?
    # On vérifie que les messages de log attendus ont été émis
    expected_log_calls = [
        mocker.call(f"📂 Lecture du dataset source : {expected_input_csv}"),
        mocker.call("⚖️ Génération d’un dataset équilibré (10 000 échantillons)..."),
        mocker.call(f"✅ Dataset d’origine créé ({len(fake_balanced_df)} échantillons)"),
        mocker.call(f"💾 Fichier sauvegardé dans {expected_output_csv}")
    ]
    mock_log_info.assert_has_calls(expected_log_calls)