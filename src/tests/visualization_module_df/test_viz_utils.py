import pytest
import pandas as pd
import pandas.testing as pdt
import numpy as np
from unittest.mock import MagicMock, call

# Importer les fonctions à tester
from src.visualization_module_df import viz_utils

# --- Tests pour la logique pure (analyze_prediction_errors) ---

def test_analyze_prediction_errors_nominal():
    """
    Test 1: Vérifie que la fonction identifie
    correctement les erreurs.
    """
    # L'argument 'pipeline' n'est pas utilisé dans la fonction,
    # mais il est dans la signature, donc on le passe à None.
    pipeline = None 
    y_pred = np.array([1, 2, 3, 4, 5])
    y_true = np.array([1, 0, 3, 4, 0]) # Erreurs aux indices 1 et 4
    
    df_erreurs = viz_utils.analyze_prediction_errors(pipeline, y_pred, y_true)
    
    # Données attendues
    expected_data = {
        'index': [1, 4],
        'predicted': [2, 5],
        'true': [0, 0]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Utiliser l'utilitaire de test de pandas
    pdt.assert_frame_equal(df_erreurs, expected_df)

def test_analyze_prediction_errors_no_errors():
    """
    Test 2: Vérifie que la fonction retourne un
    DataFrame vide s'il n'y a pas d'erreurs.
    """
    pipeline = None
    y_pred = np.array([1, 2, 3])
    y_true = np.array([1, 2, 3])
    
    df_erreurs = viz_utils.analyze_prediction_errors(pipeline, y_pred, y_true)
    
    # Données attendues (DataFrame vide avec les bonnes colonnes)
    expected_df = pd.DataFrame(columns=['index', 'predicted', 'true'])
    
    pdt.assert_frame_equal(df_erreurs, expected_df, check_dtype=False)

# --- Fixtures pour mocker les plots ---

@pytest.fixture
def mock_plotting(mocker):
    """
    Fixture centrale pour mocker toutes les dépendances
    de plotting et de système de fichiers.
    """
    # Mocker sklearn
    mock_cm = mocker.patch("src.visualization_module_df.confusion_matrix")
    mock_cmd_class = mocker.patch("src.visualization_module_df.ConfusionMatrixDisplay")
    
    # Mocker matplotlib
    mock_plt_subplots = mocker.patch("src.visualization_module_df.plt.subplots")
    mock_plt_title = mocker.patch("src.visualization_module_df.plt.title")
    mock_plt_savefig = mocker.patch("src.visualization_module_df.plt.savefig")
    mock_plt_close = mocker.patch("src.visualization_module_df.plt.close")
    mock_plt_figure = mocker.patch("src.visualization_module_df.plt.figure")
    mock_plt_xlabel = mocker.patch("src.visualization_module_df.plt.xlabel")
    mock_plt_ylabel = mocker.patch("src.visualization_module_df.plt.ylabel")
    mock_plt_tight_layout = mocker.patch("src.visualization_module_df.plt.tight_layout")
    
    # Mocker seaborn
    mock_sns_histplot = mocker.patch("src.visualization_module_df.sns.histplot")
    
    # Mocker os
    mock_os_makedirs = mocker.patch("src.visualization_module_df.os.makedirs")
    
    # Simuler le retour de plt.subplots()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt_subplots.return_value = (mock_fig, mock_ax)
    
    # Simuler le retour de ConfusionMatrixDisplay
    mock_disp = MagicMock()
    mock_cmd_class.return_value = mock_disp

    # Retourner les mocks pour que les tests puissent les inspecter
    return {
        "confusion_matrix": mock_cm,
        "ConfusionMatrixDisplay": mock_cmd_class,
        "disp": mock_disp,
        "subplots": mock_plt_subplots,
        "ax": mock_ax,
        "title": mock_plt_title,
        "savefig": mock_plt_savefig,
        "close": mock_plt_close,
        "figure": mock_plt_figure,
        "xlabel": mock_plt_xlabel,
        "ylabel": mock_plt_ylabel,
        "tight_layout": mock_plt_tight_layout,
        "histplot": mock_sns_histplot,
        "makedirs": mock_os_makedirs
    }

# --- Tests pour plot_confusion_matrix ---

def test_plot_confusion_matrix_with_save(mock_plotting):
    """
    Test 3: Vérifie que la CM est calculée, plotée,
    et SAUVEGARDÉE.
    """
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    labels = [0, 1]
    title = "My Test Matrix"
    save_path = "/fake/dir/plot.png"
    
    viz_utils.plot_confusion_matrix(y_true, y_pred, labels=labels, title=title, save_path=save_path)
    
    # 1. Vérifier les appels à sklearn
    mock_plotting["confusion_matrix"].assert_called_once_with(y_true, y_pred, labels=labels)
    mock_plotting["ConfusionMatrixDisplay"].assert_called_once()
    
    # 2. Vérifier les appels de plotting
    mock_plotting["subplots"].assert_called_once_with(figsize=(10, 8))
    mock_plotting["disp"].plot.assert_called_once_with(
        ax=mock_plotting["ax"], cmap='Blues', values_format='d'
    )
    mock_plotting["title"].assert_called_once_with(title)
    
    # 3. Vérifier les appels de sauvegarde
    mock_plotting["makedirs"].assert_called_once_with("/fake/dir", exist_ok=True)
    mock_plotting["savefig"].assert_called_once_with(save_path, bbox_inches='tight')
    
    # 4. Vérifier que le plot est fermé
    mock_plotting["close"].assert_called_once()

def test_plot_confusion_matrix_without_save(mock_plotting):
    """
    Test 4: Vérifie que la CM est plotée mais
    PAS sauvegardée si save_path est None.
    """
    viz_utils.plot_confusion_matrix(y_true=[0], y_pred=[0], save_path=None)
    
    # Vérifier que les appels de sauvegarde ne sont PAS faits
    mock_plotting["makedirs"].assert_not_called()
    mock_plotting["savefig"].assert_not_called()
    
    # Vérifier que le plot est quand même fermé
    mock_plotting["close"].assert_called_once()

# --- Tests pour plot_prediction_distribution ---

def test_plot_prediction_distribution(mock_plotting):
    """
    Test 5: Vérifie que l'histogramme de distribution
    est ploté et sauvegardé.
    """
    proba_df = pd.DataFrame({'confidence': [0.1, 0.5, 0.9]})
    model_name = "TestModel"
    save_dir = "/fake/plots"
    expected_save_path = "/fake/plots/distribution_confiance_TestModel.png"
    
    viz_utils.plot_prediction_distribution(proba_df, model_name, save_dir=save_dir)
    
    # 1. Vérifier la création du dossier
    mock_plotting["makedirs"].assert_called_once_with(save_dir, exist_ok=True)
    
    # 2. Vérifier les appels de plotting
    mock_plotting["figure"].assert_called_once_with(figsize=(10, 6))
    mock_plotting["histplot"].assert_called_once_with(
        proba_df['confidence'], bins=30, kde=True, color='blue'
    )
    mock_plotting["title"].assert_called_once_with("Distribution des confiances - TestModel")
    mock_plotting["xlabel"].assert_called_once_with("Confiance")
    mock_plotting["ylabel"].assert_called_once_with("Nombre de prédictions")
    mock_plotting["tight_layout"].assert_called_once()
    
    # 3. Vérifier la sauvegarde
    mock_plotting["savefig"].assert_called_once_with(expected_save_path)
    
    # 4. Vérifier que le plot est fermé
    mock_plotting["close"].assert_called_once()