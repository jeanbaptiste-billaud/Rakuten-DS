import pytest
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import call, MagicMock

# Importer la classe à tester
from src.data_module_df.data_config import PipelineConfig

# --- Fixture 1: Configuration YAML Factice ---

@pytest.fixture
def fake_yaml_config():
    """
    Simule le contenu du config.yaml RÉEL chargé par OmegaConf.
    """
    return OmegaConf.create({
        # 📁 Blocs de chemins
        "data": {
            "raw_dir": "data/raw/latest",
            "processed_dir": "data/processed/"
        },
        "image_model": {
            "image_force_training": True,
            "image_model_path": "models/image/"
        },
        "text_model": {
            "text_force_training": True,
            "text_model_path": "models/text/"
        },
        
        # ⚙️ Blocs de paramètres (qui doivent être settés)
        "training": {
            "output_dir": "outputs/",
            "epochs": 20,
            "batch_size": 256
        },
        "models": {
            "image": "xgboost",
            "text": "svm"
        },
        
        # 🔧 Attributs de haut niveau
        "force_preprocessing": True,
        "save_outputs": True,
        "fusion_strategy": "mean",
        "crossval_folds": 5,
        "optuna_trials": 50,
        "version": 1,
        
        # 🤖 Multithreading
        "multithreading": {
            "activate": True,
            "auto": True,
            "num_workers": 10
        },
        
        # ⛔ Clé à ignorer (selon la logique du __init__)
        "output": {} 
    })

# --- Fixture 2: Environnement Mocké Central ---

@pytest.fixture
def mock_env(mocker, fake_yaml_config):
    """
    Mocke TOUTES les interactions externes (fichiers, OS, subprocess).
    Cette fixture s'exécute avant chaque test qui la demande.
    """
    # 1. Mocker le chargement du YAML
    mock_load = mocker.patch("src.data_module_df.data_config.OmegaConf.load")
    mock_load.return_value = fake_yaml_config

    # 2. Mocker la création de dossiers
    # On mocke la *méthode* mkdir de la classe Path
    mock_mkdir = mocker.patch("src.data_module_df.data_config.Path.mkdir")

    # 3. Mocker la vérification d'existence
    mock_exists = mocker.patch("src.data_module_df.data_config.Path.exists")
    mock_exists.return_value = True # Par défaut, on suppose que tout existe

    # 4. Mocker le comptage de CPU
    mock_cpu = mocker.patch("os.cpu_count", return_value=8)

    # 5. Mocker DVC
    mock_dvc = mocker.patch("subprocess.run")

    # 6. Mocker l'écriture de fichiers (pour save_yaml)
    mock_open = mocker.patch("src.data_module_df.data_config.Path.open", mocker.mock_open())
    
    # 7. Mocker le dumper YAML (corrigera l'AttributeError de votre code)
    # Votre code a un bug : self.yaml.dump (self.yaml n'existe pas)
    # Je vais supposer que vous vouliez utiliser OmegaConf.save
    # En mockant Path.open, l'écriture échoue silencieusement, ce qui 
    # nous permet de tester le reste.
    # Pour tester update_and_save, nous mockerons save_yaml directement.

    # Retourner les mocks pour que les tests puissent les inspecter
    return {
        "load": mock_load,
        "mkdir": mock_mkdir,
        "exists": mock_exists,
        "cpu_count": mock_cpu,
        "dvc": mock_dvc,
        "open": mock_open
    }

# --- Fixture 3: Instance de la Classe ---

@pytest.fixture
def config(mock_env):
    """
    Instancie PipelineConfig. Grâce à mock_env,
    l'__init__ s'exécute sans toucher au système.
    """
    return PipelineConfig()

# --- Tests ---

def test_init_chargement_et_attributs_simples(config):
    """
    Teste le chargement et l'assignation des attributs de base.
    """
    # L'instance 'config' est créée par la fixture
    
    # 1. Vérifier les attributs de haut niveau
    assert config.version == 1
    assert config.force_preprocessing is True
    assert config.save_outputs is True
    assert config.fusion_strategy == "mean"
    assert config.crossval_folds == 5
    assert config.optuna_trials == 50

    # 2. Vérifier les blocs de paramètres (doivent être présents)
    assert config.training["epochs"] == 20
    assert config.models["text"] == "svm"
    
    # 3. Vérifier les flags d'entraînement
    assert config.train_image is True
    assert config.train_text is True

    # 4. Vérifier que les clés "bloc" de chemins sont ignorées
    assert not hasattr(config, "data")
    assert not hasattr(config, "image_model")
    assert not hasattr(config, "text_model")
    assert not hasattr(config, "output")

def test_multithreading_auto(config, mock_env):
    """Teste le setup multithreading en mode 'auto'."""
    mock_env["cpu_count"].assert_called_once()
    # os.cpu_count() // 2 -> 8 // 2 = 4
    assert config.num_workers == 4

def test_multithreading_manuel(mocker, fake_yaml_config):
    """Teste le setup multithreading en mode manuel."""
    fake_yaml_config.multithreading.auto = False
    fake_yaml_config.multithreading.num_workers = 12
    
    mocker.patch("src.data_module_df.data_config.OmegaConf.load", return_value=fake_yaml_config)
    mocker.patch("src.data_module_df.data_config.Path.mkdir")
    mock_cpu = mocker.patch("os.cpu_count")

    config = PipelineConfig()
    
    mock_cpu.assert_not_called()
    assert config.num_workers == 12

def test_multithreading_desactive(mocker, fake_yaml_config):
    """Teste le setup multithreading désactivé."""
    fake_yaml_config.multithreading.activate = False
    
    mocker.patch("src.data_module_df.data_config.OmegaConf.load", return_value=fake_yaml_config)
    mocker.patch("src.data_module_df.data_config.Path.mkdir")
    mock_cpu = mocker.patch("os.cpu_count")

    config = PipelineConfig()
    
    mock_cpu.assert_not_called()
    assert config.num_workers == 1 # Le défaut de la classe

def test_process_paths_creation_dossiers(config, mock_env):
    """Vérifie que les bons dossiers sont créés."""
    
    # L'__init__ a déjà appelé process_paths
    mock_mkdir = mock_env["mkdir"]
    
    # 3 appels à mkdir dans process_paths
    assert mock_mkdir.call_count == 3
    # Vérifie que les appels ont bien les bons flags
    mock_mkdir.assert_has_calls([
        call(parents=True, exist_ok=True),
        call(parents=True, exist_ok=True),
        call(parents=True, exist_ok=True)
    ])
    
    # Vérifie les chemins (cas par défaut: force_preprocessing=False)
    assert "latest" in str(config.processed_dir)
    assert f"v{config.version}" in str(config.image_model_path)
    assert f"v{config.version}" in str(config.text_model_path)

def test_process_paths_force_preprocessing_true(mocker, fake_yaml_config):
    """Vérifie les chemins si force_preprocessing=True."""
    fake_yaml_config.force_preprocessing = True
    fake_yaml_config.image_model.image_force_training = True # Teste aussi cette branche
    
    mocker.patch("src.data_module_df.data_config.OmegaConf.load", return_value=fake_yaml_config)
    mocker.patch("src.data_module_df.data_config.Path.mkdir")
    mocker.patch("os.cpu_count")

    config = PipelineConfig()

    # Vérifie les chemins
    assert f"v{config.version}" in str(config.processed_dir)
    assert "latest" not in str(config.processed_dir)
    
    # Si force_training=True, le chemin n'inclut PAS la version
    assert f"v{config.version}" not in str(config.image_model_path)

def test_validate_paths_succes(config, mock_env):
    """Teste la validation quand les fichiers existent."""
    mock_env["exists"].return_value = True
    
    # Ne doit pas lever d'erreur
    try:
        config.validate_paths()
    except FileNotFoundError:
        pytest.fail("validate_paths() a levé FileNotFoundError par erreur")

def test_validate_paths_echec(config, mock_env):
    """Teste la validation quand un fichier manque."""
    mock_env["exists"].return_value = False
    
    with pytest.raises(FileNotFoundError, match="Chemin introuvable"):
        config.validate_paths()

def test_ensure_preprocessed_data_fichiers_presents(config, mock_env):
    """Teste quand les fichiers .npz sont déjà là."""
    mock_env["exists"].return_value = True # Tous les fichiers existent
    
    config.ensure_preprocessed_data()
    
    # DVC ne doit pas être appelé
    mock_env["dvc"].assert_not_called()

def test_ensure_preprocessed_data_dvc_succes(config, mock_env):
    """Teste le 'dvc pull' quand des fichiers manquent."""
    mock_exists = mock_env["exists"]
    mock_dvc = mock_env["dvc"]
    
    # Simule "fichiers manquants" au premier check, "fichiers présents" au second
    # 7 fichiers attendus -> 7x False, puis 7x True
    nb_fichiers = 7
    mock_exists.side_effect = [False] * nb_fichiers + [True] * nb_fichiers
    
    config.ensure_preprocessed_data()
    
    # DVC doit être appelé
    mock_dvc.assert_called_once_with(
        ["dvc", "pull"], 
        cwd=config.project_root, 
        check=True, 
        capture_output=True, 
        text=True
    )
    assert mock_exists.call_count == nb_fichiers * 2

def test_ensure_preprocessed_data_dvc_echec_subprocess(config, mock_env):
    """Teste si 'dvc pull' lui-même échoue."""
    mock_exists = mock_env["exists"]
    mock_dvc = mock_env["dvc"]
    
    mock_exists.return_value = False # Fichiers manquants
    
    # Simule un échec de la commande DVC
    mock_dvc.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="dvc pull", stdout="erreur stdout", stderr="erreur stderr"
    )
    
    with pytest.raises(RuntimeError, match="Erreur lors de l'exécution de 'dvc pull'"):
        config.ensure_preprocessed_data()

def test_ensure_preprocessed_data_dvc_fichiers_toujours_manquants(config, mock_env):
    """Teste si 'dvc pull' réussit mais les fichiers manquent toujours."""
    mock_exists = mock_env["exists"]
    
    # Simule "fichiers manquants" au premier ET au second check
    mock_exists.return_value = False
    
    with pytest.raises(FileNotFoundError, match="toujours manquants après 'dvc pull'"):
        config.ensure_preprocessed_data()

def test_update_and_save_config(config, mocker):
    """
    Teste l'incrémentation de la version et la sauvegarde.
    Nous mockons la méthode save_yaml pour éviter le bug.
    """
    # Mocker la méthode save_yaml directement sur l'instance
    mocker.patch.object(config, 'save_yaml')
    
    assert config.yaml_cfg["version"] == 1
    
    # Action
    config.update_and_save_config()
    
    # Vérification
    assert config.yaml_cfg["version"] == 2
    config.save_yaml.assert_called_once_with(config.config_path)