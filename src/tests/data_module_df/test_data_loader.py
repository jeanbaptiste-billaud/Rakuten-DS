import pytest
import pandas as pd
import numpy as np
import pandas.testing as pdt
import numpy.testing as npt
from unittest.mock import MagicMock

# Importer le module à tester
from src.data_module_df import data_loader

# --- Fixture pour la logique pure ---

@pytest.fixture
def sample_text_df():
    """DataFrame factice pour tester la combinaison de texte."""
    return pd.DataFrame({
        "designation": ["Livre A", np.nan, "Jeu B"],
        "description": ["Description A", "Description B", np.nan]
    })

# --- Tests pour la logique pure ---

def test_combine_text_fields(sample_text_df):
    """
    Teste la fonction de logique pure 'combine_text_fields'.
    Vérifie la gestion des NaN et la concaténation.
    """
    result = data_loader.combine_text_fields(sample_text_df)
    
    # Le résultat doit être un array numpy
    assert isinstance(result, np.ndarray)
    
    # fillna('') + ' ' + fillna('') ne supprime pas les espaces
    expected = np.array([
        "Livre A Description A",
        " Description B", # Espace au début car 'designation' est NaN
        "Jeu B "          # Espace à la fin car 'description' est NaN
    ])
    
    npt.assert_array_equal(result, expected)

# --- Tests pour les fonctions I/O (Lecture) ---

def test_load_training_data(mocker):
    """
    Teste 'load_training_data' en mockant pd.read_csv.
    """
    # 1. Préparation des mocks
    fake_x = pd.DataFrame({'feature': [1, 2]})
    fake_y = pd.DataFrame({'label': [0, 1]})
    
    mock_read_csv = mocker.patch("pandas.read_csv")
    # Retourne fake_x au premier appel, fake_y au second
    mock_read_csv.side_effect = [fake_x, fake_y]
    
    # 2. Exécution
    x_df, y_df = data_loader.load_training_data("/fake/dir")
    
    # 3. Vérification
    # Vérifie que pd.read_csv a été appelé 2x avec les bons args
    assert mock_read_csv.call_count == 2
    mock_read_csv.assert_has_calls([
        mocker.call("/fake/dir/X_train.csv", index_col=0),
        mocker.call("/fake/dir/Y_train.csv", index_col=0)
    ])
    
    # Vérifie que les DataFrames retournés sont les bons
    pdt.assert_frame_equal(x_df, fake_x)
    pdt.assert_frame_equal(y_df, fake_y)

def test_load_processed_npz_success(mocker):
    """
    Teste 'load_processed_npz' quand le fichier existe.
    Nous mockons os.path.exists et np.load.
    """
    # 1. Préparation
    fake_data = {"key": "value"}
    data_dir = "/fake/dir"
    name = "my_data"
    expected_path = "/fake/dir/processed/my_data.npz"
    
    mocker.patch("os.path.exists", return_value=True)
    
    # Simuler la structure complexe de chargement :
    # np.load() -> NpzFile['arr_0'] -> ndarray.item() -> data
    mock_load = mocker.patch("numpy.load")
    mock_npz_file = MagicMock()
    mock_load.return_value = mock_npz_file
    # Simule NpzFile['arr_0'].item()
    mock_npz_file.__getitem__.return_value.item.return_value = fake_data

    # 2. Exécution
    result = data_loader.load_processed_npz(name, data_dir)
    
    # 3. Vérification
    mock_load.assert_called_once_with(expected_path, allow_pickle=True)
    assert result == fake_data

def test_load_processed_npz_not_found(mocker):
    """
    Teste 'load_processed_npz' quand le fichier n'existe pas.
    """
    # 1. Préparation
    data_dir = "/fake/dir"
    name = "my_data"
    expected_path = "/fake/dir/processed/my_data.npz"
    
    mocker.patch("os.path.exists", return_value=False)
    mock_load = mocker.patch("numpy.load") # Pour vérifier qu'il n'est pas appelé
    
    # 2. Exécution & Vérification
    with pytest.raises(FileNotFoundError, match=f"Fichier non trouvé : {expected_path}"):
        data_loader.load_processed_npz(name, data_dir)
        
    mock_load.assert_not_called()

# --- Tests pour les fonctions I/O (Écriture) ---

def test_save_processed_npz(mocker):
    """
    Teste 'save_processed_npz' en mockant os.makedirs et np.savez.
    Vérifie que les bons chemins et les bonnes données sont passés.
    """
    # 1. Préparation
    mock_makedirs = mocker.patch("os.makedirs")
    mock_savez = mocker.patch("numpy.savez")
    
    fake_data = {"a": 1, "b": 2}
    data_dir = "/fake/dir"
    name = "my_save"
    
    expected_dir = "/fake/dir/processed"
    expected_path = "/fake/dir/processed/my_save.npz"
    # Le code sauvegarde les données sous une clé spécifique
    expected_data_to_save = {"my_save_": fake_data}
    
    # 2. Exécution
    data_loader.save_processed_npz(fake_data, name, data_dir)
    
    # 3. Vérification
    mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
    mock_savez.assert_called_once_with(expected_path, **expected_data_to_save)

# --- Test I/O (Aller-Retour) ---

def test_save_and_load_index_split_roundtrip(tmp_path):
    """
    Teste 'save_index_split' et 'load_index_split' ensemble.
    C'est la meilleure façon de tester une sauvegarde/chargement compatible.
    
    'tmp_path' est une fixture pytest qui fournit un dossier temporaire (Pathlib).
    """
    # 1. Préparation
    data_dir = str(tmp_path) # Convertir Path en string
    indices_to_save = {
        'train_idx': np.array([1, 5, 9]),
        'val_idx': np.array([2, 6]),
        'test_idx': np.array([3, 4, 7, 8])
    }
    
    # 2. Exécution (Sauvegarde)
    data_loader.save_index_split(indices_to_save, data_dir)
    
    # 3. Vérification (fichier créé)
    expected_file = tmp_path / "processed" / "indices_split.npz"
    assert expected_file.exists(), "Le fichier .npz n'a pas été créé au bon endroit"
    
    # 4. Exécution (Chargement)
    loaded_indices = data_loader.load_index_split(data_dir)
    
    # 5. Vérification (données)
    assert isinstance(loaded_indices, dict)
    assert loaded_indices.keys() == indices_to_save.keys()
    
    npt.assert_array_equal(loaded_indices['train_idx'], indices_to_save['train_idx'])
    npt.assert_array_equal(loaded_indices['val_idx'], indices_to_save['val_idx'])
    npt.assert_array_equal(loaded_indices['test_idx'], indices_to_save['test_idx'])