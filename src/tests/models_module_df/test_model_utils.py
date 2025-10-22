import pytest
import numpy as np
import numpy.testing as npt
import yaml
from unittest.mock import MagicMock, call

# Importer le module à tester
from src.models_module_df import model_utils

# --- Tests pour load_model_configs ---

@pytest.fixture
def mock_yaml_loading(mocker):
    """Fixture pour simuler la lecture d'un fichier YAML."""
    # Simuler le contenu du fichier YAML
    fake_yaml_content = """
    xgboost:
      param1: 10
    neural_net:
      param2: 20
    """
    fake_config_dict = {'xgboost': {'param1': 10}, 'neural_net': {'param2': 20}}
    
    # 1. Mocker os.path.join pour qu'il retourne un chemin prévisible
    mocker.patch("os.path.join", return_value="/fake/path/train_image.yaml")
    
    # 2. Mocker la fonction 'open'
    # mock_open lit 'fake_yaml_content'
    mocker.patch("builtins.open", mocker.mock_open(read_data=fake_yaml_content))
    
    # 3. Mocker yaml.safe_load pour qu'il retourne notre dictionnaire
    mock_safe_load = mocker.patch("yaml.safe_load", return_value=fake_config_dict)
    
    return mock_safe_load

def test_load_model_configs_success(mock_yaml_loading):
    """
    Test 1: Vérifie le cas nominal où le YAML est chargé
    correctement et contient tous les modèles attendus.
    """
    configs = model_utils.load_model_configs()
    
    # Vérifie que yaml.safe_load a été appelé
    mock_yaml_loading.assert_called_once()
    
    # Vérifie que le dictionnaire retourné est correct
    assert "xgboost" in configs
    assert configs["neural_net"]["param2"] == 20
    assert len(configs) == 2

def test_load_model_configs_missing_model(mocker, mock_yaml_loading):
    """
    Test 2: Vérifie qu'un avertissement est imprimé si un modèle
    attendu est manquant dans le fichier de configuration.
    """
    # Remplacer le retour du mock pour ce test
    mock_yaml_loading.return_value = {'xgboost': {'param1': 10}} # 'neural_net' est manquant
    
    # Mocker 'print' pour capturer sa sortie
    mock_print = mocker.patch("builtins.print")
    
    configs = model_utils.load_model_configs()
    
    # Le dictionnaire partiel est quand même retourné
    assert "xgboost" in configs
    assert "neural_net" not in configs
    
    # Vérifier que le bon avertissement a été imprimé
    mock_print.assert_called_with("⚠️ Modèles manquants dans la configuration : {'neural_net'}")

def test_load_model_configs_file_not_found(mocker):
    """
    Test 3: Vérifie que la fonction gère une FileNotFoundError,
    imprime une erreur et retourne un dictionnaire vide.
    """
    # 1. Mocker os.path.join
    mocker.patch("os.path.join", return_value="/fake/path/train_image.yaml")
    
    # 2. Mocker 'open' pour qu'il lève une erreur
    mocker.patch("builtins.open", side_effect=FileNotFoundError("Fichier test non trouvé"))
    
    # 3. Mocker 'print' pour capturer l'erreur
    mock_print = mocker.patch("builtins.print")
    
    configs = model_utils.load_model_configs()
    
    # Vérifier que le bon message d'erreur est imprimé
    mock_print.assert_called_with("Erreur lors du chargement des configurations : Fichier test non trouvé")
    
    # Vérifier que le retour est un dictionnaire vide
    assert configs == {}

# --- Tests pour cross_validate_model ---

def test_cross_validate_model(mocker):
    """
    Test 4: Vérifie que la fonction 'cross_validate_model'
    appelle 'cross_val_score' et formate correctement la sortie.
    """
    # 1. Préparation
    fake_model = MagicMock()
    fake_X = np.array([[1], [2], [3], [4]])
    fake_y = np.array([0, 0, 1, 1])
    
    # Définir les scores que 'cross_val_score' doit retourner
    fake_scores = np.array([0.8, 0.9, 0.7])
    
    # 2. Mocker la fonction de sklearn
    mock_cvs = mocker.patch(
        "src.models_module_df.model_utils.cross_val_score", 
        return_value=fake_scores
    )
    
    # 3. Exécution
    result = model_utils.cross_validate_model(
        fake_model, fake_X, fake_y, cv=3, scoring='f1'
    )
    
    # 4. Vérification
    # A-t-on appelé 'cross_val_score' avec les bons arguments ?
    mock_cvs.assert_called_once_with(
        fake_model, fake_X, fake_y, cv=3, scoring='f1'
    )
    
    # Le dictionnaire de résultats est-il correct ?
    assert result['mean_score'] == np.mean(fake_scores) # 0.8
    assert result['std_score'] == np.std(fake_scores)
    npt.assert_array_equal(result['all_scores'], fake_scores)

# --- Tests pour optimize_hyperparameters ---

def test_optimize_hyperparameters(mocker):
    """
    Test 5: Vérifie que la fonction 'optimize_hyperparameters'
    crée, lance et extrait les résultats d'une étude Optuna.
    """
    # 1. Préparation
    
    # CORRECTION ICI: Nous patchons optuna.create_study À SA SOURCE
    # car l'import se fait dans la fonction
    mock_create_study = mocker.patch("optuna.create_study") 
    
    # Créer un faux 'study object'
    mock_study = MagicMock()
    mock_study.best_value = 0.95
    mock_study.best_params = {'C': 10, 'kernel': 'rbf'}
    
    # Configurer le mock
    mock_create_study.return_value = mock_study
    
    fake_objective_fn = MagicMock(return_value=0.9)
    mock_print = mocker.patch("builtins.print")

    # 2. Exécution
    best_params = model_utils.optimize_hyperparameters(
        fake_objective_fn, 
        n_trials=25, 
        direction='maximize'
    )
    
    # 3. Vérification
    # A-t-on créé l'étude avec la bonne direction ?
    mock_create_study.assert_called_once_with(direction='maximize')
    
    # A-t-on lancé l'optimisation avec les bons paramètres ?
    mock_study.optimize.assert_called_once_with(fake_objective_fn, n_trials=25)
    
    # Les bons messages ont-ils été imprimés ?
    mock_print.assert_has_calls([
        call("✅ Meilleur score : 0.95"),
        call("🏆 Meilleurs paramètres : {'C': 10, 'kernel': 'rbf'}")
    ])
    
    # La fonction a-t-elle retourné les meilleurs paramètres ?
    assert best_params == {'C': 10, 'kernel': 'rbf'}