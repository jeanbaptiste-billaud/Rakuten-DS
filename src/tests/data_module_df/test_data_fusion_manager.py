import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
# Importer la classe à tester
from src.data_module_df.data_fusion_manager import FusionManager


# --- Fixtures pour les probabilités ---

@pytest.fixture
def prob_fixtures():
    """
    Crée deux jeux de probabilités (2 échantillons, 3 classes).
    - text_probs : Confiance en Classe 0 (0.8) et Classe 1 (0.9)
    - image_probs: Confiance en Classe 2 (0.7) et Classe 2 (0.5)
    Les modèles sont en désaccord, ce qui est parfait pour tester la fusion.
    """
    text_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.9, 0.0]
    ])
    
    image_probs = np.array([
        [0.1, 0.2, 0.7],
        [0.1, 0.4, 0.5]
    ])
    return text_probs, image_probs

# --- Tests pour la méthode fuse_predictions ---

def test_fuse_strategy_mean(prob_fixtures):
    """Teste la fusion par moyenne."""
    text_p, image_p = prob_fixtures
    
    # Calcul attendu : (A + B) / 2
    expected_fused = np.array([
        [(0.8 + 0.1) / 2, (0.1 + 0.2) / 2, (0.1 + 0.7) / 2], # [0.45, 0.15, 0.40]
        [(0.1 + 0.1) / 2, (0.9 + 0.4) / 2, (0.0 + 0.5) / 2]  # [0.10, 0.65, 0.25]
    ])
    
    fused = FusionManager.fuse_predictions(text_p, image_p, strategy='mean')
    
    npt.assert_allclose(fused, expected_fused)

def test_fuse_strategy_product(prob_fixtures):
    """Teste la fusion par produit (avec normalisation)."""
    text_p, image_p = prob_fixtures
    
    # Calcul attendu : (A * B) / sum(A * B)
    # Normalisation (chaque ligne doit sommer à 1)
    fused_sum_0 = 0.08 + 0.02 + 0.07 # 0.17
    fused_sum_1 = 0.01 + 0.36 + 0.00 # 0.37
    
    expected_fused = np.array([
        [0.08 / fused_sum_0, 0.02 / fused_sum_0, 0.07 / fused_sum_0],
        [0.01 / fused_sum_1, 0.36 / fused_sum_1, 0.00 / fused_sum_1]
    ])
    
    fused = FusionManager.fuse_predictions(text_p, image_p, strategy='product')
    
    npt.assert_allclose(fused, expected_fused)

def test_fuse_strategy_weighted(prob_fixtures):
    """Teste la fusion par pondération fixe (0.6 Texte, 0.4 Image)."""
    text_p, image_p = prob_fixtures
    
    # Calcul attendu : A * 0.6 + B * 0.4
    expected_fused = (text_p * 0.6) + (image_p * 0.4)
    
    # Ligne 0 : [0.48+0.04, 0.06+0.08, 0.06+0.28] = [0.52, 0.14, 0.34]
    # Ligne 1 : [0.06+0.04, 0.54+0.16, 0.00+0.20] = [0.10, 0.70, 0.20]
    
    fused = FusionManager.fuse_predictions(text_p, image_p, strategy='weighted')
    
    npt.assert_allclose(fused, expected_fused)

def test_fuse_strategy_confidence_weighted(prob_fixtures):
    """Teste la fusion pondérée par la confiance."""
    text_p, image_p = prob_fixtures
    
    # --- Calcul manuel ---
    # Ligne 0:
    text_conf_0 = 0.8
    image_conf_0 = 0.7
    total_0 = 1.5
    w_text_0 = text_conf_0 / total_0   # 0.8 / 1.5 = 0.5333...
    w_image_0 = image_conf_0 / total_0 # 0.7 / 1.5 = 0.4666...
    fused_0 = text_p[0] * w_text_0 + image_p[0] * w_image_0
    
    # Ligne 1:
    text_conf_1 = 0.9
    image_conf_1 = 0.5
    total_1 = 1.4
    w_text_1 = text_conf_1 / total_1   # 0.9 / 1.4 = 0.6428...
    w_image_1 = image_conf_1 / total_1 # 0.5 / 1.4 = 0.3571...
    fused_1 = text_p[1] * w_text_1 + image_p[1] * w_image_1
    
    expected_fused = np.array([fused_0, fused_1])
    # --- Fin calcul ---
    
    fused = FusionManager.fuse_predictions(text_p, image_p, strategy='confidence_weighted')
    
    npt.assert_allclose(fused, expected_fused)

def test_fuse_invalid_strategy(prob_fixtures):
    """Teste qu'une stratégie inconnue lève une erreur ValueError."""
    text_p, image_p = prob_fixtures
    
    with pytest.raises(ValueError, match="strategy 'median' not supported"):
        FusionManager.fuse_predictions(text_p, image_p, strategy='median')

# --- Fixtures pour le rapport ---

@pytest.fixture
def report_data():
    """Crée les 7 arrays nécessaires pour le rapport (3 échantillons)."""
    y_true =       np.array([10, 20, 10])
    text_preds =   np.array([10, 10, 10]) # Correct, Faux, Correct
    image_preds =  np.array([10, 20, 20]) # Correct, Correct, Faux
    fused_preds =  np.array([10, 20, 10]) # Correct, Correct, Correct (Fusion a aidé échantillon 1)
    
    text_probs = np.array([
        [0.8, 0.1, 0.1], # Conf 0.8
        [0.7, 0.2, 0.1], # Conf 0.7
        [0.6, 0.3, 0.1]  # Conf 0.6
    ])
    image_probs = np.array([
        [0.6, 0.3, 0.1], # Conf 0.6
        [0.1, 0.1, 0.8], # Conf 0.8 (Classe 20)
        [0.1, 0.1, 0.8]  # Conf 0.8
    ])
    fused_probs = np.array([
        [0.9, 0.05, 0.05], # Conf 0.9
        [0.1, 0.8, 0.1],  # Conf 0.8
        [0.7, 0.2, 0.1]   # Conf 0.7
    ])
    # Correction des prédictions pour correspondre aux probabilités
    # (np.argmax(probs, axis=1) ne correspond pas aux classes 10, 20)
    # Les prédictions sont les classes (10, 20), pas les indices (0, 1)
    # Le code create_combined_report ne se soucie pas de cela, il compare
    # juste les arrays. Les données ci-dessus sont valides.
    
    return y_true, text_preds, image_preds, fused_preds, text_probs, image_probs, fused_probs
    

# --- Tests pour la méthode create_combined_report ---

def test_create_combined_report(report_data):
    """Teste la création du DataFrame de rapport."""
    y_true, text_preds, image_preds, fused_preds, text_probs, image_probs, fused_probs = report_data
    
    df = FusionManager.create_combined_report(
        y_true, text_preds, image_preds, fused_preds, text_probs, image_probs, fused_probs
    )
    
    # 1. Vérifier le type et la taille
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(y_true) # Doit avoir 3 lignes
    
    # 2. Définir le DataFrame attendu
    expected_data = {
        'true': [10, 20, 10],
        'text_pred': [10, 10, 10],
        'image_pred': [10, 20, 20],
        'fused_pred': [10, 20, 10],
        'text_conf': [0.8, 0.7, 0.6],       # np.max(text_probs[i])
        'image_conf': [0.6, 0.8, 0.8],      # np.max(image_probs[i])
        'fused_conf': [0.9, 0.8, 0.7],      # np.max(fused_probs[i])
        'agree_text_image': [True, False, False], # text_preds[i] == image_preds[i]
        'correct_fusion': [True, True, True]    # fused_preds[i] == y_true[i]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # 3. Comparer les DataFrames
    pdt.assert_frame_equal(df, expected_df)
